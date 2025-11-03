# ===============================================
# 💬 Banking Intention Router (Phase 2 — v3.1 Soft Anti-Hallucination Defense)
# -----------------------------------------------
# ✅ Unified with Phase 1 (Whisper model)
# ✅ Dual Voice: static-first vs dynamic Edge-TTS
# ✅ Clean defense against hallucinated or empty transcriptions
# ✅ Returns polite fallback reply instead of HTTP 400
# ===============================================

import os, csv, json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import joblib
from pydub import AudioSegment
from intention.entity_extractor import extract_entities
from intention.anti_hallucination import defend   # 🧠 new import

# =========================================================
# 🔹 Global toggles (env defaults)
# =========================================================
ENV_DIAG = os.getenv("DIAG", "0") == "1"
ENV_REPLY_MODE = os.getenv("REPLY_MODE", "0") == "1"  # False=static-first, True=dynamic

def mk_dlog(enabled: bool):
    def _dlog(*args, **kwargs):
        if enabled:
            print(*args, **kwargs)
    return _dlog

# =========================================================
# 🔹 Optional Edge-TTS synthesis
# =========================================================
try:
    from intention.edge_tts import synthesize_tts
    TTS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ TTS module import failed: {e}")
    TTS_AVAILABLE = False

# =========================================================
# 🔹 Paths & Model Loading
# =========================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENT_DIR = os.path.join(BASE_DIR, "intention")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DOCS_DIR   = os.path.join(STATIC_DIR, "docs")
TTS_DIR    = os.path.join(STATIC_DIR, "tts")

ARCHIVE_DIR = os.path.join(BASE_DIR, "local_persistent", "intention_archive")
os.makedirs(os.path.join(ARCHIVE_DIR, "wavs"), exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

VEC_PATH = os.path.join(INTENT_DIR, "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(INTENT_DIR, "intent_classifier.pkl")
DM_PATH  = os.path.join(INTENT_DIR, "domain_model.json")

if not (os.path.exists(VEC_PATH) and os.path.exists(CLF_PATH)):
    raise RuntimeError("❌ Intent model files not found — run train_intent_model.py first!")

vectorizer   = joblib.load(VEC_PATH)
classifier   = joblib.load(CLF_PATH)
domain_model = json.load(open(DM_PATH, "r", encoding="utf-8")) if os.path.exists(DM_PATH) else {}

router = APIRouter(tags=["Bank Intention Handler"])

# =========================================================
# 🔹 Response Schema
# =========================================================
class IntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: Dict[str, Any] = {}
    action: str
    reply_text: Optional[str] = None
    pdf_url: Optional[str] = None
    voice_url: Optional[str] = None
    static_voice_file: Optional[str] = None
    balance_amount: Optional[str] = None

# =========================================================
# 🔹 Core Helpers
# =========================================================
def classify_text(text: str) -> Dict[str, Any]:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")
    X = vectorizer.transform([text])
    pred = classifier.predict(X)[0]
    probs = classifier.predict_proba(X)[0]
    conf = float(max(probs))
    entities = extract_entities(text)
    return {"intent": pred, "confidence": conf, "entities": entities, "action": "auto"}

def get_domain_action(intent: str) -> Dict[str, Any]:
    if intent in domain_model:
        return domain_model[intent]
    return {
        "description": "Unknown intent",
        "action": "voice_reply",
        "reply_text": "Уучлаарай, таны хүсэлтийг ойлгосонгүй."
    }

def append_metadata(user_id: str, fname: str, text: str, result: Dict[str, Any], dlog):
    meta_path = os.path.join(ARCHIVE_DIR, "metadata.csv")
    header = ["record_id","user_id","file_name","created_at","text",
              "intent","confidence","entities","action"]
    exists = os.path.exists(meta_path)
    record_id = 1
    if exists:
        with open(meta_path, "r", encoding="utf-8") as m:
            lines = m.readlines()
            if len(lines) > 1:
                record_id = len(lines)
    with open(meta_path, "a", encoding="utf-8", newline="") as m:
        w = csv.writer(m)
        if not exists:
            w.writerow(header)
        w.writerow([
            record_id, user_id, fname,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text, result.get("intent",""), result.get("confidence",0.0),
            json.dumps(result.get("entities", {}), ensure_ascii=False),
            result.get("action","")
        ])
    dlog(f"🗂️ Metadata updated for record {record_id}: {text}")

# =========================================================
# 🔹 Voice Resolver (Dual Mode — global REPLY_MODE only)
# =========================================================
def resolve_voice_simple(intent_key: str,
                         intent_data: Dict[str, Any],
                         reply_mode_dynamic: bool,
                         dlog) -> Optional[str]:
    reply_text = intent_data.get("reply_text", "")
    static_rel = intent_data.get("static_voice_file")

    def abs_from_rel(rel: str) -> str:
        if rel.startswith("/static/"):
            return os.path.join(BASE_DIR, rel.lstrip("/"))
        return os.path.join(STATIC_DIR, rel)

    # --- Dynamic Mode ---
    if reply_mode_dynamic:
        if not TTS_AVAILABLE:
            dlog(f"[{intent_key}] Dynamic requested but TTS unavailable.")
            return None
        dlog(f"[{intent_key}] 🎙️ Dynamic mode → Edge-TTS synthesis")
        rel_path = synthesize_tts(
            reply_text,
            voice="mn-MN-YesuiNeural",
            output_dir="static/tts",
            basename=f"{intent_key}_reply"
        )
        dlog(f"[{intent_key}] ✅ Synthesized → {rel_path}")
        return f"/{rel_path}"

    # --- Static Mode (default) ---
    if static_rel:
        static_abs = abs_from_rel(static_rel.strip("/"))
        if os.path.exists(static_abs):
            voice_url = f"/static/{static_rel.strip('/')}"
            dlog(f"[{intent_key}] ✅ Using static studio WAV: {voice_url}")
            return voice_url
        else:
            dlog(f"[{intent_key}] ⚠️ Static file missing → fallback to TTS")

    # --- Fallback to TTS if static missing ---
    if not TTS_AVAILABLE:
        dlog(f"[{intent_key}] ❌ No TTS available.")
        return None
    rel_path = synthesize_tts(
        reply_text,
        voice="mn-MN-YesuiNeural",
        output_dir="static/tts",
        basename=f"{intent_key}_reply"
    )
    dlog(f"[{intent_key}] 🌀 Static fallback → synthesized {rel_path}")
    return f"/{rel_path}"

# =========================================================
# 🔹 Helper: Always load balance text file if present
# =========================================================
def load_balance_text(intent_data: Dict[str, Any], dlog) -> Optional[str]:
    balance_cfg = intent_data.get("balance_amount")
    if not balance_cfg or not balance_cfg.endswith(".txt"):
        return None
    abs_path = os.path.join(BASE_DIR, "static", balance_cfg)
    if not os.path.exists(abs_path):
        dlog(f"⚠️ Balance file missing: {abs_path}")
        return None
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        dlog(f"⚠️ Failed reading balance file: {e}")
        return None

# =========================================================
# 🔹 Endpoint 1 — Text Classification
# =========================================================
@router.post("/classify", response_model=IntentResponse)
def classify_intent(
    text: str = Form(...),
    DIAG_: Optional[int] = Form(None),
    REPLY_MODE_: Optional[int] = Form(None)
):
    req_diag = ENV_DIAG if DIAG_ is None else (str(DIAG_) == "1")
    dlog = mk_dlog(req_diag)
    reply_mode_dynamic = ENV_REPLY_MODE if REPLY_MODE_ is None else (str(REPLY_MODE_) == "1")

    result = classify_text(text)
    domain_action = get_domain_action(result["intent"])
    result.update(domain_action)

    if "static_voice_file" in domain_action:
        result["static_voice_file"] = domain_action["static_voice_file"]

    if result["action"] == "voice_reply":
        result["voice_url"] = resolve_voice_simple(
            result["intent"], domain_action, reply_mode_dynamic, dlog
        )
        balance_text = load_balance_text(domain_action, dlog)
        if balance_text:
            result["balance_amount"] = balance_text

    if result["action"] in ("pdf_reply", "file_reply"):
        result["pdf_url"] = domain_action.get("pdf_url")

    dlog(f"🧠 Intent: {result['intent']} (conf={result['confidence']:.2f}) → {result['action']}")
    return result

# =========================================================
# 🔹 Endpoint 2 — Voice Input (Soft Anti-Hallucination)
# =========================================================
@router.post("/voice_intent", response_model=IntentResponse)
async def classify_from_voice(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    DIAG_: Optional[int] = Form(None),
    REPLY_MODE_: Optional[int] = Form(None)
):
    req_diag = ENV_DIAG if DIAG_ is None else (str(DIAG_) == "1")
    dlog = mk_dlog(req_diag)
    reply_mode_dynamic = ENV_REPLY_MODE if REPLY_MODE_ is None else (str(REPLY_MODE_) == "1")

    contents = await file.read()
    wav_dir = os.path.join(ARCHIVE_DIR, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    fname = f"int{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    fpath = os.path.join(wav_dir, fname)

    try:
        fmt = "wav"
        ext = os.path.splitext(file.filename or "")[1].lower()
        if "webm" in ext: fmt = "webm"
        elif "mp4" in ext or "m4a" in ext: fmt = "mp4"
        elif "ogg" in ext: fmt = "ogg"
        tmp = fpath + ".tmp"
        with open(tmp, "wb") as t: t.write(contents)
        audio = AudioSegment.from_file(tmp, format=fmt)
        os.remove(tmp)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(fpath, format="wav")
        dlog(f"✅ Archived clean WAV → {fpath}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {e}")

    try:
        from app import model
        if model is None:
            raise HTTPException(status_code=503, detail="Whisper model not loaded yet")
        segments, info = model.transcribe(
            fpath, language="mn", task="transcribe",
            vad_filter=True, beam_size=1, best_of=1,
            temperature=0.0, word_timestamps=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        dlog(f"🎧 Transcribed → {text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {e}")

    # 🧠 --- Anti-hallucination defense ---
    clean_text = defend(text, dlog)
    if not clean_text or len(clean_text) < 4:
        dlog("🚫 Hallucinated or invalid text detected → returning gentle fallback reply.")
        polite_reply = {
            "intent": "unknown",
            "confidence": 0.0,
            "action": "voice_reply",
            "reply_text": "Уучлаарай, таны яриаг ойлгосонгүй.",
            "voice_url": "/static/tts/unknown_reply.wav",
            "static_voice_file": "/static/tts/unknown_reply.wav",
        }
        append_metadata(user_id, fname, text, polite_reply, dlog)
        return polite_reply

    text = clean_text
    result = classify_text(text)
    domain_action = get_domain_action(result["intent"])
    result.update(domain_action)

    if "static_voice_file" in domain_action:
        result["static_voice_file"] = domain_action["static_voice_file"]

    if result["action"] == "voice_reply":
        result["voice_url"] = resolve_voice_simple(
            result["intent"], domain_action, reply_mode_dynamic, dlog
        )
        balance_text = load_balance_text(domain_action, dlog)
        if balance_text:
            result["balance_amount"] = balance_text

    if result["action"] in ("pdf_reply", "file_reply"):
        result["pdf_url"] = domain_action.get("pdf_url")

    append_metadata(user_id, fname, text, result, dlog)
    dlog(f"🎙️ Voice → '{text}' → {result['intent']} (conf={result['confidence']:.2f})")
    return result
