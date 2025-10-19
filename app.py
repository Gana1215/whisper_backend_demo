import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import soundfile as sf  # for quick duration guard

# -------- Config --------
MODEL_DIR = os.getenv("MODEL_DIR", "models/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")              # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8") # cpu: int8/float32; gpu: float16/int8_float16

# Optional guards for the Hobby plan (adjust if you upgrade later)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))              # 30 sec

app = FastAPI(title="Mongolian Whisper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # lock down to your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model load on startup (once) --------
model: WhisperModel | None = None

@app.on_event("startup")
def load_model():
    global model
    # Removed any PyTorch .bin existence checks — we load CT2/faster-whisper directly
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model folder not found: {MODEL_DIR}")

    # These limits help stay under 512 MB on the Hobby plan
    model = WhisperModel(
        MODEL_DIR,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=1,   # keep small to reduce RAM spikes
        num_workers=1    # single worker to avoid duplicating model in memory
    )
    print(f"✅ Model loaded: {MODEL_DIR} | device: {DEVICE} | compute: {COMPUTE_TYPE}")

@app.get("/")
def health():
    return {"ok": True, "msg": "Mongolian Whisper API is running."}

class TranscribeResult(BaseModel):
    text: str
    language: str | None = None
    duration_sec: float | None = None

@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Basic content-type check
    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    # Size guard (if client provided length header; not always present)
    cl = getattr(file, "size", None)
    if cl and cl > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    # Save to temp file
    try:
        suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # Optional duration guard to avoid OOM on long audio
    try:
        audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        dur = len(audio) / float(sr)
        if dur > MAX_DURATION_SEC:
            raise HTTPException(
                status_code=413,
                detail=f"Audio too long ({dur:.1f}s). Limit is {MAX_DURATION_SEC:.0f}s on the Hobby plan."
            )
    except HTTPException:
        # forward size/duration specific errors
        raise
    except Exception:
        # If we can’t read duration, proceed — decoder will still try
        dur = None

    # Inference (greedy decode to minimize memory)
    try:
        segments, info = model.transcribe(
            tmp_path,
            language="mn",          # Mongolian
            task="transcribe",      # not "translate"
            vad_filter=True,
            beam_size=1,            # ↓ memory vs. beam search
            best_of=1,              # ↓ memory vs. best-of
            temperature=0.0,
            word_timestamps=False
        )
        text = "".join(seg.text for seg in segments).strip()
        return TranscribeResult(text=text, language=info.language, duration_sec=info.duration if hasattr(info, "duration") else dur)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
