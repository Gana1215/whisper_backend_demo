# ===============================================
# 🎙️ Mongolian Whisper API (FastAPI + Faster-Whisper)
# Compatible with Python 3.9+
# ===============================================

import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import soundfile as sf  # for quick duration guard
import logging

# -------- Logging Setup --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------- Config --------
MODEL_DIR = os.getenv("MODEL_DIR", "models/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")               # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # cpu: int8/float32; gpu: float16/int8_float16
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))               # 30 sec

# -------- FastAPI setup --------
app = FastAPI(title="Mongolian Whisper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down to your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model holder --------
model: Optional[WhisperModel] = None


# -------- Model load on startup (once) --------
@app.on_event("startup")
def load_model():
    """Load Faster-Whisper model once at startup."""
    global model

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model folder not found: {MODEL_DIR}")

    logging.info(f"🔄 Loading model from {MODEL_DIR} (device={DEVICE}, compute={COMPUTE_TYPE})")

    model = WhisperModel(
        MODEL_DIR,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=1,   # minimize RAM spikes
        num_workers=1
    )

    logging.info(f"✅ Model loaded successfully: {MODEL_DIR} | device={DEVICE} | compute={COMPUTE_TYPE}")


# -------- Health check --------
@app.get("/")
def health():
    return {"ok": True, "msg": "Mongolian Whisper API is running."}


# -------- Response schema --------
class TranscribeResult(BaseModel):
    text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None


# -------- Main endpoint --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(file: UploadFile = File(...)):
    """Receive an audio file and return transcription."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # --- MIME guard ---
    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    # --- Size guard ---
    cl = getattr(file, "size", None)
    if cl and cl > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    # --- Save to temp file ---
    try:
        suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        logging.info(f"📥 Received file: {file.filename} -> {tmp_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # --- Duration guard ---
    try:
        audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        dur = len(audio) / float(sr)
        if dur > MAX_DURATION_SEC:
            raise HTTPException(
                status_code=413,
                detail=f"Audio too long ({dur:.1f}s). Limit is {MAX_DURATION_SEC:.0f}s on the Hobby plan."
            )
    except HTTPException:
        raise
    except Exception:
        dur = None  # if unreadable, continue anyway

    # --- Inference ---
    try:
        logging.info(f"🎧 Starting transcription (duration={dur:.1f}s)")
        segments, info = model.transcribe(
            tmp_path,
            language="mn",
            task="transcribe",
            vad_filter=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            word_timestamps=False
        )
        text = "".join(seg.text for seg in segments).strip()
        logging.info(f"✅ Transcription complete: {len(text)} chars")
        return TranscribeResult(
            text=text,
            language=getattr(info, "language", "mn"),
            duration_sec=getattr(info, "duration", dur)
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        try:
            os.remove(tmp_path)
            logging.info(f"🧹 Temp file removed: {tmp_path}")
        except Exception:
            pass


# -------- Local entrypoint (for dev / Render) --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render injects $PORT automatically
    logging.info(f"🚀 Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
