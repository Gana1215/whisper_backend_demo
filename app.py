import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

# -------- Config --------
MODEL_DIR = os.getenv("MODEL_DIR", "models/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")         # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # cpu: int8 / float32; gpu: float16/int8_float16

app = FastAPI(title="Mongolian Whisper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # replace with your Vercel domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model load on startup (once) --------
model: WhisperModel | None = None

@app.on_event("startup")
def load_model():
    global model
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    # Warm a tiny dummy to allocate kernels (optional)
    print("✅ Model loaded:", MODEL_DIR, "| device:", DEVICE, "| compute:", COMPUTE_TYPE)

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

    # Only allow audio content types commonly used
    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    # Save to a temp file (faster-whisper needs a path or bytes-like)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1] or ".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # Inference
    try:
        # Force Mongolian transcription (no translation)
        # Adjust decoding options to your preference
        segments, info = model.transcribe(
            tmp_path,
            language="mn",          # Mongolian
            task="transcribe",      # not "translate"
            vad_filter=True,        # helps with silence/noise
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        text = "".join(seg.text for seg in segments).strip()
        return TranscribeResult(text=text, language=info.language, duration_sec=info.duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
