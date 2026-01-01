# ===============================================
# 💬 Banking Intention Router
#    Phase 2 — v3.1.2 (Dynamic Clarify + Yesui TTS Cache + Direct-Intent Buttons)
# -----------------------------------------------
# ✅ Keeps v3.1.1 logic fully intact (dynamic clarify, Yesui TTS cache)
# ✅ Adds 1:1 direct-intent handling for clarify-button clicks
# ✅ Clarify MP3 auto-cache via mn-MN-YesuiNeural (edge-tts)
# ✅ Fallback to /static/tts/fallback_clarify.mp3 on failure
# ===============================================

import os, csv, json, time, re, subprocess, shlex
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from fastapi import APIRouter, UploadFile, File, Form, Request   # ✅ PATCH: add Request
from pydantic import BaseModel
import joblib
from pydub import AudioSegment
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from intention.entity_extractor import extract_entities

# =========================================================
# 🎧 Smart Static Voice Reply — MP3 → WAV fallback
# =========================================================
def get_static_voice_path(static_file: str, static_dir: str = "static") -> Optional[str]:
    base = (static_file or "").replace(".wav", "").replace(".mp3", "").strip()
    if not base:
        print("⚠️ Missing audio file spec")
        return None
    mp3_path = os.path.join(static_dir, f"{base}.mp3")
    wav_path = os.path.join(static_dir, f"{base}.wav")
    if os.path.exists(mp3_path):
        print(f"🎵 Using MP3 version → {mp3_path}")
        return f"{base}.mp3"
    if os.path.exists(wav_path):
        print(f"🎵 Using WAV fallback → {wav_path}")
        return f"{base}.wav"
    print(f"⚠️ Missing audio file for {static_file}")
    return None

# =========================================================
# 🔹 Global toggles
# =========================================================
ENV_DIAG = os.getenv("DIAG", "0") == "1"
CONF_THRESH = float(os.getenv("INTENT_CONF_THRESHOLD", "0.25"))
FALLBACK_K = int(os.getenv("INTENT_FALLBACK_K", "3"))
YESUI_VOICE = "mn-MN-YesuiNeural"

def mk_dlog(enabled: bool):
    def _dlog(*a, **k):
        if enabled: print(*a, **k)
    return _dlog

# =========================================================
# 🔹 Paths & Model Loading
# =========================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENT_DIR = os.path.join(BASE_DIR, "intention")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TTS_DIR    = os.path.join(STATIC_DIR, "tts")
COREBANK_CSV = os.path.join(STATIC_DIR, "corebank_reply.csv")
ARCHIVE_DIR  = os.path.join(BASE_DIR, "local_persistent", "intention_archive")
os.makedirs(os.path.join(ARCHIVE_DIR, "wavs"), exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

VEC_PATH = os.path.join(INTENT_DIR, "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(INTENT_DIR, "intent_classifier.pkl")
DM_PATH  = os.path.join(INTENT_DIR, "domain_model.json")
CSV_PATH = os.path.join(INTENT_DIR, "intents.csv")

vectorizer   = joblib.load(VEC_PATH)
classifier   = joblib.load(CLF_PATH)
domain_model = json.load(open(DM_PATH, "r", encoding="utf-8")) if os.path.exists(DM_PATH) else {}

# =========================================================
# 🔹 Load training corpus for cosine fallback
# =========================================================
_train_texts, _train_labels = [], []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    header = f.readline()
    if "text" not in header or "intent" not in header:
        f.seek(0)
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            _train_texts.append(row[0].strip())
            _train_labels.append(row[1].strip().lower())
_TRAIN_X = vectorizer.transform(_train_texts)
_TRAIN_Y = np.array(_train_labels)

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
    corebank_data: Optional[Dict[str, str]] = None
    base_intent: Optional[str] = None
    base_confidence: Optional[float] = None
    fallback_used: Optional[bool] = None
    intent_choices: Optional[List[str]] = None
    clarify_keyword: Optional[str] = None

# =========================================================
# 🔹 /voice_intent (PATCHED ONLY HERE)
# =========================================================
@router.post("/voice_intent", response_model=IntentResponse)
async def classify_from_voice(
    request: Request,                      # ✅ PATCH
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    dlog = mk_dlog(ENV_DIAG)
    wav_dir = os.path.join(ARCHIVE_DIR, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    fname = f"int{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    tmp = f"{fname}.tmp"

    with open(tmp, "wb") as t:
        t.write(await file.read())

    audio = AudioSegment.from_file(tmp)
    os.remove(tmp)

    path = os.path.join(wav_dir, fname)
    audio.set_frame_rate(44100).set_channels(1).set_sample_width(2).export(path, format="wav")

    # ✅ PATCH: SAFE model access (no circular import)
    model = request.app.state.asr_model
    if model is None:
        raise RuntimeError("ASR model not loaded")

    segs, _ = model.transcribe(path, language="mn", task="transcribe", vad_filter=True)
    text = "".join(s.text for s in segs).strip()
    dlog(f"🎧 Transcribed → {text}")

    # ---- everything below is UNCHANGED ----
    result = classify_text(text, dlog)
    pred_intent, prob = result["intent"], result["confidence"]

    clarify = maybe_clarify_flow(text, pred_intent, prob, dlog)
    if clarify:
        return IntentResponse(
            intent="clarify",
            confidence=prob,
            entities=result.get("entities", {}),
            action=clarify["action"],
            reply_text=clarify["reply_text"],
            voice_url=clarify.get("voice_url"),
            intent_choices=clarify.get("intent_choices"),
            clarify_keyword=clarify.get("clarify_keyword"),
            base_intent=result.get("base_intent"),
            base_confidence=result.get("base_confidence"),
            fallback_used=result.get("fallback_used"),
        )

    domain_action = get_domain_action(pred_intent)
    result.update(domain_action)
    cb = attach_corebank_data(pred_intent, dlog)
    if cb: result["corebank_data"] = cb
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(pred_intent, cb).strip() if cb else ""
    result["reply_text"] = f"{base_reply}\n{secure_part}" if secure_part else base_reply or "Таны хүсэлтийг хүлээн авлаа."
    svf = (domain_action.get("static_voice_file") or "").strip().lstrip("/")
    resolved = get_static_voice_path(svf, STATIC_DIR)
    result["voice_url"] = f"/static/{resolved}" if resolved else None
    return IntentResponse(**result)
