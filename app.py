# ===============================================
# 🎙️ Mongolian Whisper API — Phase 2 (Final Unified v2.2.2)
# ---------------------------------------------------
# ✅ Phase2/2A ROUTES UNCHANGED: /dataset + /intent + static/tts
# ✅ CT2 (faster-whisper) engine
# ✅ GOLDEN PATCH: Render-safe CT2 load by local snapshot (NO tokenizer overwrite)
# ✅ PROD PATCH: Anti-stutter decoding (beam=5, repetition penalty, no-repeat ngram, warm prompt)
# ✅ Keeps SAME /transcribe response schema for BankAI frontend
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

# ✅ CT2 / faster-whisper (NO PyTorch HF)
from faster_whisper import WhisperModel

# ✅ Render-safe CT2 download
from huggingface_hub import snapshot_download

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

# ✅ Set this to your BetterGolden CT2 repo (FP16 or INT8)
HF_MODEL = os.getenv("HF_MODEL", "gana1215/MN_Whisper_Base_CT2")

DEVICE = os.getenv("DEVICE", "cpu")               # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # "int8" / "int8_float16" / "float16"

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "local_persistent/record_archive"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))
DIAG = os.getenv("DIAG", "0") == "1"

print("✅ Environment configuration loaded:")
print(f"   HF_MODEL        → {HF_MODEL} (CT2 repo/path)")
print(f"   DEVICE          → {DEVICE}")
print(f"   COMPUTE_TYPE    → {COMPUTE_TYPE}")
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

# ===============================================
# ✅ PATCH: Make intent router boot-safe if domain_model.json is empty/invalid
# ===============================================
def _ensure_valid_json_file(path: str):
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        if not txt.strip():
            raise ValueError("empty json")
        import json as _json
        _json.loads(txt)
    except Exception as e:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("{}\n")
            print(f"🧩 Patched invalid JSON → {path} ({e})")
        except Exception as e2:
            print(f"⚠️ Could not patch invalid JSON file: {path} ({e2})")

_ensure_valid_json_file(os.path.join(BASE_DIR, "intention", "domain_model.json"))
_ensure_valid_json_file(os.path.join(BASE_DIR, "intention", "domain_model_v2.json"))
_ensure_valid_json_file(os.path.join(BASE_DIR, "intention", "domain_model_final.json"))

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

from intention.intent_router import router as intent_router
app.include_router(intent_router, prefix="/intent")
print("✅ Mounted /intent routes successfully")

# -------- Static mounts --------
app.mount("/record_archive", StaticFiles(directory=ARCHIVE_DIR), name="record_archive")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------- Whisper model (CT2) --------
model: Optional[WhisperModel] = None

@app.on_event("startup")
def load_model():
    global model
    log_memory("Before loading model")
    try:
        dlog(f"🔄 Loading CT2 model: {HF_MODEL}")

        # ============================================
        # ✅ GOLDEN PATCH (Render-safe CT2 load)
        # - Download CT2 repo to persistent disk
        # - DO NOT overwrite tokenizer.json (keeps 1200 Golden banking tokens)
        # ============================================
        local_ct2_dir = os.path.join(BASE_DIR, "local_persistent", "ct2_model")
        os.makedirs(local_ct2_dir, exist_ok=True)

        snapshot_download(
            repo_id=HF_MODEL,
            local_dir=local_ct2_dir,
            local_dir_use_symlinks=False,
        )

        # ✅ Golden integrity check
        tok_path = os.path.join(local_ct2_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise RuntimeError("tokenizer.json missing in CT2 folder — Golden model pack incomplete")
        print(f"✅ Golden tokenizer preserved: {tok_path} ({os.path.getsize(tok_path)} bytes)")

        model = WhisperModel(
            local_ct2_dir,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            local_files_only=True,
        )

        app.state.asr_model = model
        dlog(f"✅ CT2 model loaded successfully (device={DEVICE}, compute_type={COMPUTE_TYPE})")
    except Exception as e:
        logging.error(f"❌ Failed to load CT2 model: {e}")
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")
    ct = (file.content_type or "").lower()
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {ct}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    # -------- Decode to AudioSegment --------
    try:
        fmt = "wav"
        ext = os.path.splitext(file.filename or "")[1].lower()
        if "webm" in ct or ext == ".webm":
            fmt = "webm"
        elif "mp4" in ct or "m4a" in ct or ext in [".mp4", ".m4a", ".aac"]:
            fmt = "mp4"
        elif "ogg" in ct or ext == ".ogg":
            fmt = "ogg"

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

    # -------- Export 16k mono WAV for CT2 --------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav16:
        audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_16k.export(tmp_wav16.name, format="wav")
        wav_path = tmp_wav16.name

    try:
        t0 = time.perf_counter()

        # ✅ PROD PATCH (anti-stutter + Golden accuracy)
        segments, info = model.transcribe(
            wav_path,
            language="mn",            # ✅ Correct for CT2/faster-whisper
            beam_size=5,              # ✅ Golden accuracy
            vad_filter=True,
            repetition_penalty=1.2,   # ✅ Prevent stutter
            no_repeat_ngram_size=3,   # ✅ Reduce loops
            condition_on_previous_text=False,  # ✅ Big anti-repeat win on segmented decode
            initial_prompt="Банк, данс, үлдэгдэл, гүйлгээ, шилжүүлэг, карт",
        )

        text = "".join(seg.text for seg in segments).strip()
        text = " ".join(text.split())  # ✅ remove accidental double spaces

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_memory("After transcription")
        dlog(f"✅ CT2 inference done ({elapsed_ms:.1f} ms)")

        return TranscribeResult(
            user_text=text,
            language="mn",
            duration_sec=dur,
            time_ms=elapsed_ms,
            playback_path=None,
        )
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            dlog(f"🧹 Temp removed: {wav_path}")

# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Starting server on port {port} (DIAG={'ON' if DIAG else 'OFF'})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info" if DIAG else "warning")
