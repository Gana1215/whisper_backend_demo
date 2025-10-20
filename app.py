# ===============================================
# 🎙️ Mongolian Whisper API (FastAPI + Faster-Whisper)
# ✅ Robust decoding (pydub) + Memory Usage Logger (DIAG toggle)
# Compatible with Python 3.9+
# ===============================================

import os
import tempfile
import psutil
import logging
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import soundfile as sf
from pydub import AudioSegment   # ✅ Added

# -------- Config --------
MODEL_DIR = os.getenv("MODEL_DIR", "models/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))              # 30 sec
DIAG = os.getenv("DIAG", "0") == "1"  # 🔧 Diagnostics toggle: 1 = verbose logs

# -------- Logging Setup --------
log_level = logging.INFO if DIAG else logging.WARNING
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

def dlog(msg: str):
    """Conditional diagnostic logger."""
    if DIAG:
        logging.info(msg)

def log_memory(label=""):
    """Logs current memory usage in MB to logs."""
    if DIAG:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logging.info(f"💾 [{label}] Memory usage: {mem_mb:.2f} MB")

# -------- FastAPI setup --------
app = FastAPI(title="Mongolian Whisper API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model holder --------
model: Optional[WhisperModel] = None

# -------- Model load on startup --------
@app.on_event("startup")
def load_model():
    """Load Faster-Whisper model once at startup."""
    global model
    log_memory("Before loading model")

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model folder not found: {MODEL_DIR}")

    dlog(f"🔄 Loading model from {MODEL_DIR} (device={DEVICE}, compute={COMPUTE_TYPE})")

    model = WhisperModel(
        MODEL_DIR,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=1,
        num_workers=1
    )

    log_memory("After loading model")
    dlog(f"✅ Model loaded successfully: {MODEL_DIR} | device={DEVICE} | compute={COMPUTE_TYPE}")

# -------- Health check --------
@app.get("/")
def health():
    return {"ok": True, "msg": "Mongolian Whisper API is running."}

# -------- Response schema --------
class TranscribeResult(BaseModel):
    text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    time_ms: Optional[float] = None

# -------- Main endpoint --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(file: UploadFile = File(...)):
    """Receive an audio file and return transcription."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")

    # --- MIME guard (allow audio/* or unknown) ---
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {ct}")

    # --- Size guard ---
    cl = getattr(file, "size", None)
    if cl and cl > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    # --- Save upload to temp file ---
    try:
        suffix = os.path.splitext(file.filename or "")[-1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            src_path = tmp.name
        dlog(f"📥 Received file: {file.filename} -> {src_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # --- Decode with pydub to WAV ---
    wav_path = None
    dur = None
    try:
        audio = AudioSegment.from_file(src_path)
        dur = len(audio) / 1000.0  # ms -> s
        if dur > MAX_DURATION_SEC:
            raise HTTPException(
                status_code=413,
                detail=f"Audio too long ({dur:.1f}s). Limit is {MAX_DURATION_SEC:.0f}s."
            )
        wav_path = src_path + ".wav"
        audio.export(wav_path, format="wav")
        dlog(f"🎵 Audio converted to WAV | Duration={dur:.2f}s")
    except HTTPException:
        raise
    except Exception as e:
        logging.warning(f"⚠️ pydub decode failed ({e}); trying fallback")
        try:
            data, sr = sf.read(src_path, dtype="float32", always_2d=False)
            dur = len(data) / float(sr)
            wav_path = src_path
        except Exception as e2:
            logging.error(f"❌ Could not decode audio: {e2}")
            raise HTTPException(status_code=400, detail="Unsupported or corrupted audio format.")

    # --- Inference ---
    try:
        human_dur = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "unknown"
        dlog(f"🎧 Starting transcription (duration={human_dur})")

        t0 = time.perf_counter()
        segments, info = model.transcribe(
            wav_path,
            language="mn",
            task="transcribe",
            vad_filter=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            word_timestamps=False
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = "".join(seg.text for seg in segments).strip()

        log_memory("After transcription")
        dlog(f"✅ Transcription complete: {len(text)} chars | time={elapsed_ms:.1f} ms")

        return TranscribeResult(
            text=text,
            language=getattr(info, "language", "mn"),
            duration_sec=getattr(info, "duration", dur),
            time_ms=elapsed_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        for p in (wav_path, src_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    dlog(f"🧹 Temp file removed: {p}")
            except Exception:
                pass

# -------- Local entrypoint --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Starting server on port {port} (DIAG={'ON' if DIAG else 'OFF'})")
    uvicorn.run("app:app", host="0.0.0.0", port=port,
                log_level="info" if DIAG else "warning", access_log=DIAG)
