# ===============================================
# 🎙️ Mongolian Whisper API (FastAPI + Faster-Whisper)
# ✅ Handles iOS/Android/Desktop recording formats (mp4, m4a, aac, webm, wav)
# ✅ Uses temp file decode (Safari-safe)
# ✅ Works on Render + local + mobile browsers
# ✅ Unified with dataset_routes v3.5 logic
# ===============================================

import os, tempfile, psutil, logging, time, io, sys
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pydub import AudioSegment
import soundfile as sf

# -------- Load and verify environment --------
if not load_dotenv():
    print("⚠️  .env not found — using system environment variables")

required_env = ["HF_MODEL", "DEVICE", "COMPUTE_TYPE", "DATA_DIR"]
missing = [k for k in required_env if not os.getenv(k)]
if missing:
    print(f"❌ Missing required environment variables: {missing}")
    if not os.path.exists("/opt/render"):
        sys.exit(1)

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
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "local_persistent/record_archive"))

print("✅ Environment configuration loaded:")
print(f"   HF_MODEL        → {HF_MODEL}")
print(f"   DEVICE          → {DEVICE}")
print(f"   COMPUTE_TYPE    → {COMPUTE_TYPE}")
print(f"   DATA_DIR        → {DATA_DIR}")

# --- Storage folders ---
ARCHIVE_DIR = DATA_DIR if os.path.exists(DATA_DIR) else os.path.join(BASE_DIR, "local_persistent/record_archive")
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
app = FastAPI(title="Mongolian Whisper API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Include dataset routes ----
try:
    import dataset_routes
    app.include_router(dataset_routes.router)
    print("✅ dataset_routes mounted at /dataset/*")
except Exception as e:
    logging.warning(f"⚠️ dataset_routes not available ({e})")

# ---- Static mount for playback ----
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
        dlog("✅ Model loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    log_memory("After loading model")

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

# -------- Main inference endpoint --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(request: Request, file: UploadFile = File(...), device: Optional[str] = Form(None)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")
    ct = (file.content_type or "").lower()
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {ct}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    # --- Decode (multi-format, mobile-safe) ---
    dur = None
    try:
        fmt = "wav"
        ext = os.path.splitext(file.filename or "")[1].lower()

        if "webm" in ct or ext == ".webm":
            fmt = "webm"
        elif any(k in ct for k in ["mp4", "m4a", "aac"]) or ext in [".mp4", ".m4a", ".aac"]:
            fmt = "mp4"
        elif "ogg" in ct or ext == ".ogg":
            fmt = "ogg"
        elif "wav" in ct or ext == ".wav":
            fmt = "wav"

        tmp_decode = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_decode, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_decode, format=fmt)
        except Exception as e:
            logging.warning(f"⚠️ pydub decode failed ({e}); trying fallback...")
            try:
                data, sr = sf.read(tmp_decode, dtype="float32", always_2d=False)
                raw = AudioSegment(
                    data.tobytes(),
                    frame_rate=sr,
                    sample_width=4,
                    channels=1 if len(getattr(data, 'shape', [])) == 1 else data.shape[1],
                )
                audio = raw
            except Exception as e2:
                logging.error(f"❌ Fallback decode failed: {e2}")
                os.remove(tmp_decode)
                raise HTTPException(status_code=400, detail=f"Unsupported audio format ({e2})")

        os.remove(tmp_decode)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        dur = len(audio) / 1000.0
        if dur > MAX_DURATION_SEC:
            raise HTTPException(status_code=413, detail=f"Audio too long ({dur:.1f}s).")

        dlog(f"🎧 Decoded {ct} (fmt={fmt}) → {dur:.2f}s")

    except Exception as e:
        logging.error(f"❌ Decode failed ({ct}): {e}")
        raise HTTPException(status_code=400, detail=f"Unsupported audio format ({e})")

    # --- Export WAV for inference ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        audio.export(tmp_wav.name, format="wav")
        tmp_wav_path = tmp_wav.name
    dlog(f"🧩 Created temp WAV for inference → {tmp_wav_path}")

    # --- Inference ---
    try:
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            tmp_wav_path,
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
        dlog(f"✅ Inference done ({elapsed_ms:.1f} ms)")

        return TranscribeResult(
            user_text=text,
            language=getattr(info, "language", "mn"),
            duration_sec=dur,
            time_ms=elapsed_ms,
            playback_path=None
        )
    finally:
        for p in [tmp_wav_path]:
            if os.path.exists(p):
                os.remove(p)
                dlog(f"🧹 Temp removed: {p}")

# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Starting server on port {port} (DIAG={'ON' if DIAG else 'OFF'})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info" if DIAG else "warning")
