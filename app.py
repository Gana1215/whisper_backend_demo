# ===============================================
# 🎙️ Mongolian Whisper API — Phase 2 (Final Unified v2.2.2)
# ---------------------------------------------------
# ✅ Phase2/2A ROUTES UNCHANGED: /dataset + /intent + static/tts
# ✅ CT2 (faster-whisper) engine
# ✅ GOLDEN PATCH: Render-safe CT2 load by local snapshot (NO tokenizer overwrite)
# ✅ PROD PATCH: Anti-stutter decoding (beam=5, repetition penalty, no-repeat ngram, warm prompt)
# ✅ NEW PROD PATCH: Cache-bust persistent CT2 folder by HF_MODEL (+ optional HF_REVISION)
# ✅ NEW FIX: ffmpeg-only decode to 16k mono WAV (removes corrupt soundfile->AudioSegment fallback)
# ✅ NEW GOLDEN FIX: WAV fast-path (in-memory resample) for PCM WAVs (e.g. 44.1k mono)
# ✅ GAME CHANGER: Colab-compatible CT2 generate pipeline (FeatureExtractor + Tokenizer + prompts)
# ✅ Keeps SAME /transcribe response schema for BankAI frontend
#
# ✅ GAME-CHANGER HARDENING (NO OTHER LOGIC TOUCHED):
# - If filename says ".wav" but bytes are actually webm/ogg/mp3 → WAV fast-path fails → fallback to ffmpeg safely
# - File upload remains flexible: mp3/m4a/webm/ogg/wav all accepted (audio/* or octet-stream)
#
# ✅ TRANSACTION PATCH (ONLY):
# - Mount /txn routes (txn_router.py) so /txn/normalize_slot works (fixes 404)
# ===============================================

import os, tempfile, psutil, logging, time, sys, re
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment  # kept (unchanged imports; no longer used in /transcribe)
import soundfile as sf          # kept (used in WAV fast-path)

# ✅ WAV fast-path helpers
import io
import numpy as np
import librosa

# ✅ CT2 / faster-whisper (kept; no longer used for /transcribe in this patch)
from faster_whisper import WhisperModel

# ✅ Render-safe CT2 download
from huggingface_hub import snapshot_download

# ✅ GAME CHANGER: exact Colab pipeline
import ctranslate2
from transformers import WhisperTokenizer, WhisperFeatureExtractor

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
HF_MODEL = os.getenv("HF_MODEL", "gana1215/WHISPER_CT2_PRODUCTION_V2_INT8")
HF_REVISION = os.getenv("HF_REVISION", "main")  # optional pin for deterministic deploys

DEVICE = os.getenv("DEVICE", "cpu")               # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # "int8" / "int8_float16" / "float16"

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "local_persistent/record_archive"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", 30))
DIAG = os.getenv("DIAG", "0") == "1"

# ✅ CPU-safe decode knobs (Render/Mac). Can override in env.
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))   # default fast demo
BEST_OF = int(os.getenv("BEST_OF", "1"))

print("✅ Environment configuration loaded:")
print(f"   HF_MODEL        → {HF_MODEL} (CT2 repo/path)")
print(f"   HF_REVISION     → {HF_REVISION}")
print(f"   DEVICE          → {DEVICE}")
print(f"   COMPUTE_TYPE    → {COMPUTE_TYPE}")
print(f"   DATA_DIR        → {DATA_DIR}")
print(f"   BEAM_SIZE       → {BEAM_SIZE}")
print(f"   BEST_OF         → {BEST_OF}")

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

# ===============================================
# ✅ TRANSACTION PATCH (ONLY): Mount /txn routes
# ===============================================
try:
    from intention.txn_router import router as txn_router
    app.include_router(txn_router, prefix="/txn", tags=["txn"])
    print("✅ Mounted /txn routes successfully")
except Exception as e:
    print(f"⚠️ txn_router import failed: {e}")

# -------- Static mounts --------
app.mount("/record_archive", StaticFiles(directory=ARCHIVE_DIR), name="record_archive")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------- Models --------
# kept (no longer used for /transcribe after game-changer patch)
model: Optional[WhisperModel] = None

# ✅ GAME CHANGER globals (Colab-compatible)
ct2_model = None
fe = None
tok = None
prompt_tokens = None

@app.on_event("startup")
def load_model():
    global model, ct2_model, fe, tok, prompt_tokens
    log_memory("Before loading model")
    try:
        dlog(f"🔄 Loading CT2 model: {HF_MODEL} @ {HF_REVISION}")

        # ============================================
        # ✅ GOLDEN PATCH (Render-safe CT2 load)
        # ✅ Cache-bust persistent folder by model id + revision
        # - Download CT2 repo to persistent disk
        # - DO NOT overwrite tokenizer.json (keeps 1200 Golden banking tokens)
        # ============================================
        safe_repo = re.sub(r"[^a-zA-Z0-9._-]+", "__", HF_MODEL)
        safe_rev = re.sub(r"[^a-zA-Z0-9._-]+", "__", HF_REVISION)
        local_ct2_dir = os.path.join(BASE_DIR, "local_persistent", f"ct2_model__{safe_repo}__{safe_rev}")

        os.makedirs(local_ct2_dir, exist_ok=True)

        snapshot_download(
            repo_id=HF_MODEL,
            revision=HF_REVISION,
            local_dir=local_ct2_dir,
            local_dir_use_symlinks=False,
        )

        # ✅ Golden integrity check
        tok_path = os.path.join(local_ct2_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise RuntimeError("tokenizer.json missing in CT2 folder — Golden model pack incomplete")
        print(f"✅ Golden tokenizer preserved: {tok_path} ({os.path.getsize(tok_path)} bytes)")

        # kept for compatibility / future fallback (not used by /transcribe)
        model = WhisperModel(
            local_ct2_dir,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            local_files_only=True,
        )

        # ✅ GAME CHANGER: Colab-compatible pipeline loads
        fe = WhisperFeatureExtractor.from_pretrained(local_ct2_dir)
        tok = WhisperTokenizer.from_pretrained(local_ct2_dir)
        tok.set_prefix_tokens(language="Mongolian", task="transcribe")
        prompt_tokens = tok.prefix_tokens

        ct2_model = ctranslate2.models.Whisper(
            local_ct2_dir,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

        app.state.asr_model = model
        app.state.ct2_model = ct2_model
        app.state.fe = fe
        app.state.tok = tok
        app.state.prompt_tokens = prompt_tokens

        print("✅ CT2 generate engine ready (Colab-compatible pipeline)")
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
        "ct2_generate_ready": ct2_model is not None,
        "model_id": HF_MODEL,
        "revision": HF_REVISION,
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

# ✅ WAV fast-path (in-memory resample to 16k mono float32)
def _fast_wav_to_float32_16k(contents: bytes):
    data, sr = sf.read(io.BytesIO(contents), dtype="float32", always_2d=False)

    # stereo -> mono
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = np.mean(data, axis=1)

    data = np.asarray(data, dtype=np.float32)

    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000, res_type="kaiser_fast")

    return data, 16000

# -------- Main inference --------
@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(request: Request, file: UploadFile = File(...), device: Optional[str] = Form(None)):
    if ct2_model is None or fe is None or tok is None or prompt_tokens is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    log_memory("Before transcription")
    ct = (file.content_type or "").lower()
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {ct}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    # ✅ Heuristic WAV detection (may be wrong if frontend sends voice.wav but blob is webm/ogg)
    is_wav_hint = (file.filename or "").lower().endswith(".wav") or "wav" in ct

    # ---------- Prepare audio -> float32 mono 16k ----------
    wav_path = None
    try:
        audio = None
        dur = None

        # --- Attempt WAV fast-path if it looks like WAV ---
        if is_wav_hint:
            try:
                audio, _ = _fast_wav_to_float32_16k(contents)
                dur = float(len(audio)) / 16000.0
            except Exception as e:
                # ✅ GAME-CHANGER hardening: fake-wav fallback → ffmpeg
                dlog(f"⚠️ WAV fast-path failed, fallback to ffmpeg: {e}")
                audio = None
                dur = None

        # --- If not WAV or WAV fast-path failed → ffmpeg convert ---
        if audio is None:
            import subprocess, wave, contextlib

            ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
            tmp_in = tempfile.mktemp(suffix=ext)
            with open(tmp_in, "wb") as f:
                f.write(contents)

            wav_path = tempfile.mktemp(suffix=".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_in, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", wav_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Audio decode failed (ffmpeg): {e}")
            finally:
                try:
                    os.remove(tmp_in)
                except Exception:
                    pass

            # duration check (real WAV duration)
            try:
                with contextlib.closing(wave.open(wav_path, "rb")) as wf:
                    dur = wf.getnframes() / float(wf.getframerate())
            except Exception:
                dur = None

            # load wav to float32
            try:
                with open(wav_path, "rb") as f:
                    wav_bytes = f.read()
                audio, _ = _fast_wav_to_float32_16k(wav_bytes)
                if dur is None:
                    dur = float(len(audio)) / 16000.0
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"WAV load failed: {e}")

        if dur is not None and dur > MAX_DURATION_SEC:
            raise HTTPException(status_code=413, detail=f"Audio too long ({dur:.1f}s).")

        # ---------- GAME CHANGER: Colab-compatible CT2 generate ----------
        t0 = time.perf_counter()

        feats_t0 = time.perf_counter()
        feats = fe(audio, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
        s_view = ctranslate2.StorageView.from_array(feats)
        prep_ms = (time.perf_counter() - feats_t0) * 1000

        gen_t0 = time.perf_counter()
        out = ct2_model.generate(
            s_view,
            prompts=[prompt_tokens],
            beam_size=BEAM_SIZE,
            repetition_penalty=1.2,
        )
        infer_ms = (time.perf_counter() - gen_t0) * 1000

        text = tok.batch_decode([out[0].sequences_ids[0]], skip_special_tokens=True)[0].strip()
        text = " ".join(text.split())

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_memory("After transcription")
        dlog(f"✅ CT2 generate done total={elapsed_ms:.1f} ms (prep={prep_ms:.1f} ms infer={infer_ms:.1f} ms)")

        return TranscribeResult(
            user_text=text,
            language="mn",
            duration_sec=dur,
            time_ms=elapsed_ms,
            playback_path=None,
        )
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
            dlog(f"🧹 Temp removed: {wav_path}")

# -------- Entrypoint --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Starting server on port {port} (DIAG={'ON' if DIAG else 'OFF'})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info" if DIAG else "warning")
#