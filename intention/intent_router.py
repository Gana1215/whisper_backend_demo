# ===============================================
# 💬 Banking Intention Router
#    Phase 2 — Stable v2.9 (Always Reply Text)
# -----------------------------------------------
# ✅ Unified with Phase 1 Whisper backend
# ✅ Static voice replies only (no dynamic synthesis)
# ✅ CoreBank CSV auto-reload → text display only
# ✅ Never voice sensitive data like balances
# ✅ Always includes reply_text even if CB empty
# ===============================================

import os, csv, json, time
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import joblib
from pydub import AudioSegment
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from intention.entity_extractor import extract_entities

# =========================================================
# 🔹 Global toggles
# =========================================================
ENV_DIAG = os.getenv("DIAG", "0") == "1"
CONF_THRESH = float(os.getenv("INTENT_CONF_THRESHOLD", "0.25"))
FALLBACK_K = int(os.getenv("INTENT_FALLBACK_K", "3"))

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
COREBANK_CSV = os.path.join(STATIC_DIR, "corebank_reply.csv")
ARCHIVE_DIR = os.path.join(BASE_DIR, "local_persistent", "intention_archive")
os.makedirs(os.path.join(ARCHIVE_DIR, "wavs"), exist_ok=True)

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

# =========================================================
# 🔹 CoreBank CSV auto-reload system
# =========================================================
_last_load_time = 0.0
_cached_corebank: Dict[str, str] = {}

def _read_corebank_csv(path: str) -> Dict[str, str]:
    data = {}
    if not os.path.exists(path): return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "name" in row and "text" in row:
                    n = row["name"].strip().lower()
                    data[n] = row["text"].strip()
    except Exception:
        return {}
    return data

def _get_corebank_table(dlog=None) -> Dict[str, str]:
    """Auto-reloads if file modified."""
    global _last_load_time, _cached_corebank
    try:
        mtime = os.path.getmtime(COREBANK_CSV)
        if mtime != _last_load_time:
            _cached_corebank = _read_corebank_csv(COREBANK_CSV)
            _last_load_time = mtime
            dlog and dlog(f"♻️ Reloaded CoreBank CSV ({len(_cached_corebank)} entries)")
    except FileNotFoundError:
        _cached_corebank = {}
    return _cached_corebank

# Intent → CoreBank keys to fetch
INTENT_KEY_MAP = {
    "check_balance":   ["account_balance", "account_type", "customer_name"],
    "exchange_rate":   ["usd_rate", "eur_rate", "loan_rate"],
    "branch_hours":    ["branch_hours"],
    "account_details": ["account_balance", "account_type"],
    "loan_info":       ["loan_rate"],
    "customer_info":   ["customer_name"],
    "contact_support": ["branch_phone"],
}

def attach_corebank_data(intent_key: str, dlog):
    table = _get_corebank_table(dlog)
    keys = INTENT_KEY_MAP.get(intent_key, [])
    data = {k: v for k, v in table.items() if k in [x.lower() for x in keys]}
    if data:
        for k, v in data.items(): dlog and dlog(f"💾 CoreBank[{k}] → {v}")
    return data or None

# =========================================================
# 🔹 Classification core logic
# =========================================================
def _majority_vote_topk(sims: np.ndarray, k: int, dlog):
    k = max(1, min(k, sims.shape[0]))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    top_labels = _TRAIN_Y[top_idx]
    counts, best_sim = {}, {}
    for i, lab in zip(top_idx, top_labels):
        counts[lab] = counts.get(lab, 0) + 1
        s = float(sims[i])
        if lab not in best_sim or s > best_sim[lab]:
            best_sim[lab] = s
    label = max(counts, key=lambda l: (counts[l], best_sim[l]))
    dlog and dlog(f"🧭 Fallback top-{k} → {label}")
    return label, best_sim[label]

def classify_text(text: str, dlog=None):
    X = vectorizer.transform([text])
    pred = classifier.predict(X)[0]
    prob = classifier.predict_proba(X)[0].max()
    entities = extract_entities(text)
    if prob < CONF_THRESH:
        sims = cosine_similarity(X, _TRAIN_X)[0]
        voted, _ = _majority_vote_topk(sims, FALLBACK_K, dlog)
        return {"intent": voted, "confidence": prob, "entities": entities,
                "base_intent": pred, "base_confidence": prob, "fallback_used": True}
    return {"intent": pred, "confidence": prob, "entities": entities,
            "base_intent": pred, "base_confidence": prob, "fallback_used": False}

def get_domain_action(intent):
    return domain_model.get(intent, {"action": "voice_reply",
                                     "reply_text": "Уучлаарай, таны хүсэлтийг ойлгосонгүй."})

# =========================================================
# 🔹 Secure CoreBank Display Builder (never voiced)
# =========================================================
def build_secure_display(intent: str, cb: Dict[str, str]) -> str:
    """Builds privacy-safe text display (never spoken)."""
    if not cb: return ""
    if intent == "check_balance":
        txt = []
        if "account_type" in cb:
            txt.append(f"Таны дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb:
            txt.append(f"Үлдэгдэл : {cb['account_balance']}")
        return "\n".join(txt)
    if intent == "branch_hours" and "branch_hours" in cb:
        return f"Манай салбаруудын ажлын өдрүүдэд ажиллах хуваарь : {cb['branch_hours']}"
    if intent == "exchange_rate":
        parts = []
        if "usd_rate" in cb: parts.append(f"USD : {cb['usd_rate']}")
        if "eur_rate" in cb: parts.append(f"EUR : {cb['eur_rate']}")
        if "loan_rate" in cb: parts.append(f"Зээлийн хүү : {cb['loan_rate']}")
        return "  ".join(parts)
    if intent == "customer_info" and "customer_name" in cb:
        return f"Харилцагчийн нэр : {cb['customer_name']}"
    if intent == "contact_support" and "branch_phone" in cb:
        return f"☎️ Холбоо барих утас : {cb['branch_phone']}"
    if intent == "account_details":
        lines = []
        if "account_type" in cb:   lines.append(f"Дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb: lines.append(f"Үлдэгдэл : {cb['account_balance']}")
        return "\n".join(lines)
    return ""

# =========================================================
# 🔹 /classify
# =========================================================
@router.post("/classify", response_model=IntentResponse)
def classify_intent(text: str = Form(...)):
    dlog = mk_dlog(ENV_DIAG)
    result = classify_text(text, dlog)
    domain_action = get_domain_action(result["intent"])
    result.update(domain_action)

    cb = attach_corebank_data(result["intent"], dlog)
    if cb:
        result["corebank_data"] = cb

    # ✅ Always return non-empty reply_text
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(result["intent"], cb).strip() if cb else ""
    if secure_part:
        result["reply_text"] = f"{base_reply}\n{secure_part}" if base_reply else secure_part
    else:
        result["reply_text"] = base_reply or "Таны хүсэлтийг хүлээн авлаа."

    # ✅ Only set voice_url if file exists
    svf = (domain_action.get("static_voice_file") or "").strip().lstrip("/")
    voice_path = os.path.join(STATIC_DIR, svf.replace("/", os.sep))
    result["voice_url"] = f"/static/{svf}" if svf and os.path.exists(voice_path) else None

    return result

# =========================================================
# 🔹 /voice_intent
# =========================================================
@router.post("/voice_intent", response_model=IntentResponse)
async def classify_from_voice(user_id: str = Form(...), file: UploadFile = File(...)):
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

    # 🎧 Transcribe
    from app import model
    segs, _ = model.transcribe(path, language="mn", task="transcribe", vad_filter=True)
    text = "".join(s.text for s in segs).strip()
    dlog(f"🎧 Transcribed → {text}")

    result = classify_text(text, dlog)
    domain_action = get_domain_action(result["intent"])
    result.update(domain_action)

    cb = attach_corebank_data(result["intent"], dlog)
    if cb:
        result["corebank_data"] = cb

    # ✅ Always non-empty reply_text
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(result["intent"], cb).strip() if cb else ""
    if secure_part:
        result["reply_text"] = f"{base_reply}\n{secure_part}" if base_reply else secure_part
    else:
        result["reply_text"] = base_reply or "Таны хүсэлтийг хүлээн авлаа."

    # ✅ Only set voice_url if file exists
    svf = (domain_action.get("static_voice_file") or "").strip().lstrip("/")
    voice_path = os.path.join(STATIC_DIR, svf.replace("/", os.sep))
    result["voice_url"] = f"/static/{svf}" if svf and os.path.exists(voice_path) else None

    return result

# =========================================================
# 🔹 /corebank_data (debug)
# =========================================================
@router.get("/corebank_data")
def get_corebank_data():
    return _get_corebank_table()
