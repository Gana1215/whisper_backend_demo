# ===============================================
# 🎙️ Mongolian Whisper API
# Auto-fix for Git LFS pointer files + runtime model loader
# ===============================================
import os
import tempfile
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

# -------- Config --------
MODEL_DIR = os.getenv("MODEL_DIR", "models/MN_Whisper_Small_CT2")
DEVICE = os.getenv("DEVICE", "cpu")         # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # cpu: int8 / float32; gpu: float16/int8_float16

# Optional: your backup model source
MODEL_BACKUP_URL = os.getenv("MODEL_BACKUP_URL", "https://www.dropbox.com/scl/fi/xxxxx/whisper_small_ct2.zip?dl=1")

app = FastAPI(title="Mongolian Whisper API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # replace with your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- LFS Detection Helper --------
def is_lfs_pointer(file_path: str) -> bool:
    """Return True if the given file looks like a Git LFS pointer."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            return first_line.strip().startswith("version https://git-lfs.github.com/spec/")
    except Exception:
        return False

def ensure_model_files():
    """Check if real Whisper model exists, else auto-download from backup."""
    model_file = os.path.join(MODEL_DIR, "model.bin")
    alt_file = os.path.join(MODEL_DIR, "pytorch_model.bin")

    if not os.path.exists(model_file) and os.path.exists(alt_file):
        model_file = alt_file

    if not os.path.exists(model_file):
        print(f"⚠️ Model file not found at: {model_file}")
    else:
        size = os.path.getsize(model_file) / (1024 * 1024)
        if size < 100 or is_lfs_pointer(model_file):
            print(f"⚠️ Detected LFS pointer (size={size:.1f} MB). Downloading real model from backup...")
            os.makedirs(MODEL_DIR, exist_ok=True)
            zip_path = os.path.join(MODEL_DIR, "model_backup.zip")
            try:
                r = requests.get(MODEL_BACKUP_URL, stream=True)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Download complete. Extracting...")
                os.system(f"unzip -o {zip_path} -d {MODEL_DIR}")
                os.remove(zip_path)
            except Exception as e:
                print("❌ Model auto-download failed:", e)
        else:
            print(f"📏 Model file OK ({size:.1f} MB): {model_file}")

# -------- Model load on startup --------
model: WhisperModel | None = None

@app.on_event("startup")
def load_model():
    global model
    ensure_model_files()
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Model loaded:", MODEL_DIR, "| device:", DEVICE, "| compute:", COMPUTE_TYPE)

# -------- API Routes --------
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

    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1] or ".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        segments, info = model.transcribe(
            tmp_path,
            language="mn",
            task="transcribe",
            vad_filter=True,
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
