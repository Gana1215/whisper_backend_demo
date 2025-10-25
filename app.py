# ===============================================
# 🎙️ Mongolian Whisper API (FastAPI + Faster-Whisper)
# ✅ Unified storage — /record_archive/wavs only (no /uploads)
# ✅ Robust decoding (pydub + fallback)
# ✅ Works on local & Render environments
# ✅ Handles mobile uploads (iOS/Android)
# ✅ Creates permanent WAVs (usr001_*.wav) for playback
# ✅ /transcribe = inference only (NO CSV writes)
# ✅ Persistent storage support via DATA_DIR (Render/local_persistent)
# ===============================================

import os, tempfile, psutil, logging, time, datetime, io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pydub import AudioSegment
import soundfile as sf

# -------- Environment detection --------
IS_RENDER = os.path.exists("/opt/render")
BASE_DIR = "/opt/render/project/src" if IS_RENDER else os.getcwd()
os.chdir(BASE_DIR)

# -------- Config --------
HF_MODEL = os.getenv("HF_MODEL", "gana1215/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))
DIAG = os.getenv("DIAG", "0") == "1"

# --- Storage folders ---
# ✅ If DATA_DIR (Render mount) exists, use it.
# ✅ Else fallback to your local "local_persistent" directory.
DATA_DIR = os.getenv("DATA_DIR")
if DATA_DIR and os.path.exists(DATA_DIR):
    ARCHIVE_DIR = DATA_DIR
else:
    ARCHIVE_DIR = os.path.join(BASE_DIR, "local_persistent/record_archive")

ARCHIVE_WAV_DIR = os.path.join(ARCHIVE_DIR, "wavs")
os.makedirs(ARCHIVE_WAV_DIR, exist_ok=True)
print(f"📦 Dataset directory in use → {ARCHIVE_DIR}")

# -------- Logging --------
log_level = logging.INFO if DIAG else logging.WARNING
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

def dlog(msg: str):
    if DIAG:
        logging.info(msg)

def log_memory(label=""):
    if DIAG:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logging.info(f"💾 [{label}] Memory usage: {mem_mb:.2f} MB")

# -------- FastAPI setup --------
app = FastAPI(title="Mongolian Whisper API", version="1.9.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Include CSV dataset routes ----
try:
    import dataset_routes
    app.include_router(dataset_routes.router)
    print("✅ dataset_routes mounted at /dataset/*")
except Exception as e:
    logging.warning(f"⚠️ dataset_routes not available. /dataset/* endpoints disabled. {e}")

# ---- Static mounts ----
app.mount("/record_archive", StaticFiles(directory=ARCHIVE_DIR), name="record_archive")

# -------- Model --------
model: Optional[WhisperModel] = None

@app.on_event("startup")
def load_model():
    global model
    log_memory("Before loading model")
    try:
        dlog(f"🔄 Loading model: {HF_MODEL}")
        model = WhisperModel(
            HF_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=1,
            num_workers=1,
            download_root=BASE_DIR,
        )
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    log_memory("After loading model")
    dlog("✅ Model loaded successfully")

# -------- Health --------
@app.get("/")
def health():
    return {
        "ok": True,
        "msg": "Mongolian Whisper API is running.",
        "model_loaded": model is not None,
        "model_id": HF_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "archive_dir": ARCHIVE_DIR,
    }

# -------- Response schema --------
class TranscribeResult(BaseModel):
    user_text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    time_ms: Optional[float] = None
    playback_path: Optional[str] = None

# -------- Main endpoint (inference only; no CSV writes) --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    device: Optional[str] = Form(None)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")

    # --- MIME / size checks ---
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {ct}")

    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # --- Save upload to temp file ---
    try:
        suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            src_path = tmp.name
        dlog(f"📥 Uploaded file → {src_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save temp file: {e}")

    # --- Decode with pydub to WAV ---
    dur = None
    try:
        audio = AudioSegment.from_file(src_path)
        dur = len(audio) / 1000.0
        if dur > MAX_DURATION_SEC:
            raise HTTPException(status_code=413, detail=f"Audio too long ({dur:.1f}s).")
    except HTTPException:
        raise
    except Exception as e:
        logging.warning(f"⚠️ pydub decode failed ({e}); trying fallback...")
        try:
            data, sr = sf.read(src_path, dtype="float32", always_2d=False)
            dur = len(data) / float(sr)
            audio = AudioSegment.from_file(io.BytesIO(data.tobytes()), format="wav")
        except Exception as e2:
            logging.error(f"❌ Fallback decode failed: {e2}")
            raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # --- Create permanent WAV (usr001_*.wav) ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"usr001_{timestamp}.wav"
    final_path = os.path.join(ARCHIVE_WAV_DIR, final_name)

    try:
        audio.export(final_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        dlog(f"💾 Saved final WAV: {final_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export WAV: {e}")

    # --- Run inference ---
    try:
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            final_path,
            language="mn",
            task="transcribe",
            vad_filter=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            word_timestamps=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = "".join(seg.text for seg in segments).strip()
        log_memory("After transcription")
        dlog(f"✅ Inference done ({elapsed_ms:.1f} ms, text len {len(text)})")

        return TranscribeResult(
            user_text=text,
            language=getattr(info, "language", "mn"),
            duration_sec=dur,
            time_ms=elapsed_ms,
            playback_path=f"/record_archive/wavs/{final_name}",
        )

    except Exception as e:
        logging.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    finally:
        try:
            if os.path.exists(src_path):
                os.remove(src_path)
                dlog(f"🧹 Temp removed: {src_path}")
        except Exception:
            pass

# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Starting server on port {port} (DIAG={'ON' if DIAG else 'OFF'})")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info" if DIAG else "warning",
        access_log=DIAG,
    )
