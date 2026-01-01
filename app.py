# ===============================================
# 🎙️ Mongolian Whisper API — Phase 2 (Final Unified v2.2.2)
# ---------------------------------------------------
# ✅ ORIGINAL HF Whisper (Transformers) + Intent module (Phase 1 + 2 unified)
# ✅ Mounts: /dataset + /intent + static/tts
# ✅ Hugging Face model direct load (no utils/transcriber)
# ✅ Works with iOS/Android/Desktop frontends (Whisper + BankAI)
# ✅ Updated CORS: dual frontend support
# ===============================================

import os, tempfile, psutil, logging, time, sys
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment
import soundfile as sf

# ✅ NEW: Original HF Whisper (Transformers)
try:
    import torch  # optional (only needed for HF transformers version)
except Exception:
    torch = None

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 🔧 --- Ensure correct import path on Render ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"🧠 Python import path fixed: {PROJECT_ROOT}")

# -------- Load environment --------
if not load_dotenv():
    print("⚠️  .env not found — using system environment variables")

required_env = ["HF_MODEL", "DEVICE", "COMPUTE_TYPE", "DATA_DIR"]
missing = [k for k in required_env if not os.getenv(k)]
if missing:
    print(f"❌ Missing required environment variables: {missing}")
    if not os.path.exists("/opt/render"):
        sys.exit(1)

# -------- Environment setup --------
IS_RENDER = os.path.exists("/opt/render")
BASE_DIR = "/opt/render/project/src" if IS_RENDER else os.getcwd()
os.chdir(BASE_DIR)

# ✅ default now points to your ORIGINAL HF tiny
HF_MODEL = os.getenv("HF_MODEL", "gana1215/MN_Whisper_Tiny_HF")

# Keep these for compatibility with your locked envs
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "local_persistent/record_archive"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))
DIAG = os.getenv("DIAG", "0") == "1"

print("✅ Environment configuration loaded:")
print(f"   HF_MODEL        → {HF_MODEL}")
print(f"   DEVICE          → {DEVICE}")
print(f"   COMPUTE_TYPE    → {COMPUTE_TYPE} (ignored in HF mode)")
print(f"   DATA_DIR        → {DATA_DIR}")

# -------- Directories --------
ARCHIVE_DIR = DATA_DIR if os.path.exists(DATA_DIR) else os.path.join(BASE_DIR, "local_persistent/record_archive")
ARCHIVE_WAV_DIR = os.path.join(ARCHIVE_DIR, "wavs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TTS_DIR = os.path.join(STATIC_DIR, "tts")
os.makedirs(ARCHIVE_WAV_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

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
app = FastAPI(title="Mongolian Whisper API", version="2.2.2")

# ===============================================
# 🌐 CORS Setup — supports Whisper + BankAI + Local + Ngrok
# ===============================================
_default_origins = [
    "https://whisper-frontend-dhx3.onrender.com",
    "https://bankai-frontend.onrender.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_env_origins = os.getenv("FRONTEND_URLS", "")
_extra = [o.strip() for o in _env_origins.split(",") if o.strip()]
_allow_origins = list(dict.fromkeys(_default_origins + _extra))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=r"^https?://([a-z0-9-]+\.)*ngrok-free\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

print(f"🌐 CORS origins allowed → {_allow_origins}")

# -------- Include routes --------
try:
    from dataset_routes import router as dataset_router
    app.include_router(dataset_router)
    print("✅ Mounted /dataset routes successfully")
except Exception as e:
    print(f"⚠️ dataset_routes import failed: {e}")

try:
    from intention.intent_router import router as intent_router
    app.include_router(intent_router, prefix="/intent")
    print("✅ Mounted /intent routes successfully (Dual Voice Phase 2)")
except Exception as e:
    print(f"⚠️ intention router not mounted ({e})")

# -------- Static mounts --------
app.mount("/record_archive", StaticFiles(directory=ARCHIVE_DIR), name="record_archive")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------- Whisper model (HF original) --------
processor: Optional[WhisperProcessor] = None
model: Optional[WhisperForConditionalGeneration] = None

def _resolve_device() -> str:
    want = (DEVICE or "cpu").lower()
    if want in ["cuda", "gpu"] and torch.cuda.is_available():
        return "cuda"
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@app.on_event("startup")
def load_model():
    global model, processor
    log_memory("Before loading model")
    try:
        dlog(f"🔄 Loading ORIGINAL HF model: {HF_MODEL}")

        processor = WhisperProcessor.from_pretrained(HF_MODEL)

        dev = _resolve_device()
        dtype = torch.float16 if dev in ["cuda", "mps"] else torch.float32

        model = WhisperForConditionalGeneration.from_pretrained(
            HF_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(dev)
        model.eval()

        dlog(f"✅ HF model loaded successfully (device={dev}, dtype={dtype})")
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
        "model_loaded": model is not None and processor is not None,
        "model_id": HF_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,  # kept for compatibility
        "archive_dir": ARCHIVE_DIR,
        "intent_module": os.path.exists(os.path.join(BASE_DIR, "intention")),
    }

# -------- Response schema --------
class TranscribeResult(BaseModel):
    user_text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    time_ms: Optional[float] = None
    playback_path: Optional[str] = None

# -------- Main inference --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(request: Request, file: UploadFile = File(...), device: Optional[str] = Form(None)):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")
    ct = (file.content_type or "").lower()
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {ct}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        fmt = "wav"
        ext = os.path.splitext(file.filename or "")[1].lower()
        if "webm" in ct or ext == ".webm": fmt = "webm"
        elif "mp4" in ct or "m4a" in ct or ext in [".mp4", ".m4a", ".aac"]: fmt = "mp4"
        elif "ogg" in ct or ext == ".ogg": fmt = "ogg"

        tmp_decode = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_decode, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_decode, format=fmt)
        except Exception as e:
            dlog(f"⚠️ pydub decode failed ({e}); trying fallback...")
            data, sr = sf.read(tmp_decode, dtype="float32", always_2d=False)
            audio = AudioSegment(
                data.tobytes(),
                frame_rate=sr,
                sample_width=4,
                channels=1 if len(getattr(data, "shape", [])) == 1 else data.shape[1],
            )
        finally:
            os.remove(tmp_decode)

        # Keep your canonical mobile-safe output
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        dur = len(audio) / 1000.0
        if dur > MAX_DURATION_SEC:
            raise HTTPException(status_code=413, detail=f"Audio too long ({dur:.1f}s).")

        dlog(f"🎧 Decoded ({fmt}) → {dur:.2f}s")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {e}")

    # Keep your original temp wav for archival-style pipeline consistency
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        audio.export(tmp_wav.name, format="wav")
        tmp_wav_path = tmp_wav.name

    try:
        t0 = time.perf_counter()

        # ✅ HF Whisper prefers 16k mono float32
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav16:
            audio_16k.export(tmp_wav16.name, format="wav")
            tmp_wav16_path = tmp_wav16.name

        try:
            wav, sr = sf.read(tmp_wav16_path, dtype="float32", always_2d=False)
        finally:
            if os.path.exists(tmp_wav16_path):
                os.remove(tmp_wav16_path)

        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        forced = None
        try:
            forced = processor.get_decoder_prompt_ids(language="mongolian", task="transcribe")
        except Exception:
            forced = None

        with torch.no_grad():
            pred_ids = model.generate(
            inputs["input_features"],
            max_new_tokens=128,
            temperature=0.2,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            forced_decoder_ids=forced,
            num_beams=2,
)

        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log_memory("After transcription")
        dlog(f"✅ Inference done ({elapsed_ms:.1f} ms)")

        return TranscribeResult(
            user_text=text,
            language="mn",
            duration_sec=dur,
            time_ms=elapsed_ms,
            playback_path=None,
        )
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)
            dlog(f"🧹 Temp removed: {tmp_wav_path}")

# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Starting server on port {port} (DIAG={'ON' if DIAG else 'OFF'})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info" if DIAG else "warning")
