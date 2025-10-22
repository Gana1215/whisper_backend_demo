# ===============================================
# 🎙️ Mongolian Whisper API (FastAPI + Faster-Whisper)
# ✅ Robust decoding (pydub) + Memory Usage Logger (DIAG toggle)
# ✅ Works on both local & Render environments
# ✅ Handles mobile uploads (iOS/Android)
# ✅ Auto-saves uploads + permanent archive
# ✅ Serves dataset audio from /record_archive/wavs/*
# ✅ Includes CSV dataset routes (/dataset/*)
# ===============================================

import os, tempfile, psutil, logging, time, datetime
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel
import soundfile as sf
from pydub import AudioSegment

# -------- Environment detection --------
IS_RENDER = os.path.exists("/opt/render")
BASE_DIR = "/opt/render/project/src" if IS_RENDER else os.getcwd()
os.chdir(BASE_DIR)

# -------- Config --------
HF_MODEL = os.getenv("HF_MODEL", "gana1215/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))
DIAG = os.getenv("DIAG", "0") == "1"

# --- Storage folders ---
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")                 # playback (mobile)
ARCHIVE_DIR = os.path.join(BASE_DIR, "record_archive")         # permanent storage root
ARCHIVE_WAV_DIR = os.path.join(ARCHIVE_DIR, "wavs")            # <-- dataset manager expects wavs here

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_WAV_DIR, exist_ok=True)

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
app = FastAPI(title="Mongolian Whisper API", version="1.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Include CSV dataset routes (optional import safety) ----
try:
    from dataset_routes import router as dataset_router
    app.include_router(dataset_router)
    dlog("✅ dataset_routes mounted at /dataset/*")
except Exception as _e:
    logging.warning("⚠️ dataset_routes not available. /dataset/* endpoints disabled.")

# ---- Static mounts for audio playback (front-end uses these) ----
# - /uploads/<filename>.wav         (transcribe results)
# - /record_archive/wavs/<filename> (dataset audio)
app.mount("/record_archive", StaticFiles(directory=ARCHIVE_DIR), name="record_archive")

# -------- Model --------
model: Optional[WhisperModel] = None

@app.on_event("startup")
def load_model():
    """Load Whisper model from Hugging Face or local folder"""
    global model
    log_memory("Before loading model")

    try:
        dlog(f"🔄 Loading model from Hugging Face: {HF_MODEL}")
        model = WhisperModel(
            HF_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE,
            cpu_threads=1, num_workers=1, download_root=BASE_DIR
        )
    except Exception as e:
        logging.error(f"❌ Failed to load model from HF: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

    log_memory("After loading model")
    dlog("✅ Model loaded successfully")

# -------- Health --------
@app.get("/")
def health():
    return {"ok": True, "msg": "Mongolian Whisper API is running."}

# -------- Response schema --------
class TranscribeResult(BaseModel):
    user_text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    time_ms: Optional[float] = None
    playback_path: Optional[str] = None

# -------- Main endpoint --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    device: Optional[str] = Form(None)  # optional device flag ("mobile"/"desktop")
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")

    # --- MIME / size checks ---
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {ct}")
    cl = getattr(file, "size", None)
    if cl and cl > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large.")

    # --- Save upload to temp file ---
    try:
        suffix = os.path.splitext(file.filename or "")[-1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            src_path = tmp.name
        dlog(f"📥 Uploaded file: {file.filename} -> {src_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # --- Decode with pydub to WAV ---
    wav_path = None
    dur = None
    try:
        audio = AudioSegment.from_file(src_path)
        dur = len(audio) / 1000.0
        if dur > MAX_DURATION_SEC:
            raise HTTPException(status_code=413, detail=f"Audio too long ({dur:.1f}s).")
        wav_path = src_path + ".wav"
        audio.export(wav_path, format="wav")
        dlog(f"🎵 Converted to WAV ({dur:.2f}s)")
    except HTTPException:
        raise
    except Exception as e:
        logging.warning(f"⚠️ pydub decode failed ({e}); trying fallback")
        try:
            data, sr = sf.read(src_path, dtype="float32", always_2d=False)
            dur = len(data) / float(sr)
            wav_path = src_path  # already WAV-like buffer
        except Exception as e2:
            logging.error(f"❌ Decode failed: {e2}")
            raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # --- Copy to uploads & archive (dataset expects /record_archive/wavs/...) ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%Y_%H%M%S")
    playback_filename = f"wv_{timestamp}.wav"          # for /uploads
    archive_filename = f"usr001_{timestamp}.wav"       # dataset archive
    playback_path_fs = os.path.join(UPLOAD_DIR, playback_filename)
    archive_path_fs = os.path.join(ARCHIVE_WAV_DIR, archive_filename)

    try:
        with open(wav_path, "rb") as src:
            data = src.read()
        # save for immediate playback (frontend <audio src>)
        with open(playback_path_fs, "wb") as dst:
            dst.write(data)
        # save for dataset archive (used by /record_archive/wavs/<file>)
        with open(archive_path_fs, "wb") as dst:
            dst.write(data)

        src_ua = request.headers.get("user-agent", "")
        dlog(f"📱 Device flag: {device or 'unknown'} | UA: {src_ua[:80]}...")
        dlog(f"💾 Saved playback: {playback_path_fs}")
        dlog(f"💾 Saved archive:  {archive_path_fs}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to save copies: {e}")

    # --- Inference ---
    try:
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            wav_path,
            language="mn", task="transcribe",
            vad_filter=True, beam_size=1, best_of=1,
            temperature=0.0, word_timestamps=False
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = "".join(seg.text for seg in segments).strip()

        log_memory("After transcription")
        dlog(f"✅ Done ({elapsed_ms:.1f} ms, {len(text)} chars)")

        return TranscribeResult(
            user_text=text,
            language=getattr(info, "language", "mn"),
            duration_sec=getattr(info, "duration", dur),
            time_ms=elapsed_ms,
            playback_path=f"/uploads/{playback_filename}"
        )

    except Exception as e:
        logging.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        for p in (wav_path, src_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    dlog(f"🧹 Temp removed: {p}")
            except Exception:
                pass

# --- Serve playback WAVs (kept for backward compatibility) ---
@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")

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
        access_log=DIAG
    )
