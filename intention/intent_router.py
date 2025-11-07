# ===============================================================
# 💬 Banking Intention Router
#    Phase 2 — v5.1 (Smarter Canonical Edition)
# ---------------------------------------------------------------
# ✅ Unified with Phase 1 Whisper backend
# ✅ Adds Canonical Dynamic Replies (e.g., “Манай банкинд дансны талаар...”)
# ✅ Static voice replies (MP3→WAV fallback preserved)
# ✅ CoreBank CSV auto-reload → secure text display
# ✅ Avoids “Таны асуултыг ойлгосонгүй” when context is clear
# ===============================================================

import os, csv, json, time
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import joblib
from pydub import AudioSegment
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from intention.entity_extractor import extract_entities
from fastapi.responses import FileResponse
from collections import defaultdict

# =========================================================
# 🎧 Smart Static Voice Reply — MP3 → WAV fallback
# =========================================================
def get_static_voice_path(static_file: str, static_dir: str = "static") -> Optional[str]:
    base = static_file.replace(".wav", "").replace(".mp3", "")
    mp3_path = os.path.join(static_dir, f"{base}.mp3")
    wav_path = os.path.join(static_dir, f"{base}.wav")
    if os.path.exists(mp3_path):
        print(f"🎵 Using MP3 version → {mp3_path}")
        return f"{base}.mp3"
    elif os.path.exists(wav_path):
        print(f"🎵 Using WAV fallback → {wav_path}")
        return f"{base}.wav"
    else:
        print(f"⚠️ Missing audio file for {static_file}")
        return None

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
# 🔹 Canonical Intent Mapping for Dynamic Replies
# =========================================================
INTENT_KEYWORDS = {
    "check_balance": "данс",
    "account_details": "данс",
    "open_account": "данс",
    "loan_info": "зээл",
    "savings_info": "хадгаламж",
    "transfer_money": "гүйлгээ",
    "show_transactions": "хуулга",
    "exchange_rate": "ханш",
    "contact_support": "үйлчилгээ",
    "customer_info": "мэдээлэл",
    "branch_location": "салбар",
    "branch_hours": "цагийн хуваарь",
    "lost_card": "карт",
}
INTENT_GROUPS = defaultdict(list)
for i, k in INTENT_KEYWORDS.items():
    INTENT_GROUPS[k].append(i)
INTENT_LABELS = {
    "check_balance": "үлдэгдэл шалгах",
    "account_details": "дансны мэдээлэл",
    "open_account": "данс нээх",
    "loan_info": "зээлийн мэдээлэл",
    "savings_info": "хадгаламжийн үйлчилгээ",
    "transfer_money": "мөнгө шилжүүлэх",
    "show_transactions": "гүйлгээний хуулга",
    "exchange_rate": "валютын ханш",
    "contact_support": "харилцагчийн үйлчилгээ",
    "customer_info": "мэдээлэл шинэчлэх",
    "branch_location": "салбарын байршил",
    "branch_hours": "ажлын цагийн хуваарь",
    "lost_card": "карттай холбоотой үйлчилгээ",
}

def make_dynamic_bank_reply(intent: str) -> Optional[str]:
    """Return smart dynamic reply if multiple related services share same root."""
    heard_word = INTENT_KEYWORDS.get(intent)
    if not heard_word:
        return None
    related = INTENT_GROUPS[heard_word]
    if len(related) <= 1:
        return None
    suffix = "ны" if heard_word.endswith(("н", "с", "ц")) else "ийн"
    labels = [INTENT_LABELS[i] for i in related]
    options = ", ".join(labels)
    return (
        f"Манай банкинд {heard_word}-{suffix} талаар дараах үйлчилгээ байна: "
        f"{options}. Та аль үйлчилгээ авах вэ?"
    )

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
# 🔹 Secure CoreBank Display Builder
# =========================================================
def build_secure_display(intent: str, cb: Dict[str, str]) -> str:
    if not cb: return ""
    if intent == "check_balance":
        txt = []
        if "account_type" in cb:   txt.append(f"Таны дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb: txt.append(f"Үлдэгдэл : {cb['account_balance']}")
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
    if cb: result["corebank_data"] = cb
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(result["intent"], cb).strip() if cb else ""

    # ✅ Smart Canonical Reply Injection
    dynamic_reply = make_dynamic_bank_reply(result["intent"])
    if dynamic_reply:
        result["reply_text"] = dynamic_reply
    else:
        result["reply_text"] = (
            f"{base_reply}\n{secure_part}" if secure_part else base_reply or "Таны хүсэлтийг хүлээн авлаа."
        )

    svf = (domain_action.get("static_voice_file") or "").strip().lstrip("/")
    resolved = get_static_voice_path(svf, STATIC_DIR)
    result["voice_url"] = f"/static/{resolved}" if resolved else None
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

    from app import model
    segs, _ = model.transcribe(path, language="mn", task="transcribe", vad_filter=True)
    text = "".join(s.text for s in segs).strip()
    dlog(f"🎧 Transcribed → {text}")

    result = classify_text(text, dlog)
    domain_action = get_domain_action(result["intent"])
    result.update(domain_action)
    cb = attach_corebank_data(result["intent"], dlog)
    if cb: result["corebank_data"] = cb
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(result["intent"], cb).strip() if cb else ""

    # ✅ Smart Canonical Reply Injection
    dynamic_reply = make_dynamic_bank_reply(result["intent"])
    if dynamic_reply:
        result["reply_text"] = dynamic_reply
    else:
        result["reply_text"] = (
            f"{base_reply}\n{secure_part}" if secure_part else base_reply or "Таны хүсэлтийг хүлээн авлаа."
        )

    svf = (domain_action.get("static_voice_file") or "").strip().lstrip("/")
    resolved = get_static_voice_path(svf, STATIC_DIR)
    result["voice_url"] = f"/static/{resolved}" if resolved else None
    return result

# =========================================================
# 🔹 /corebank_data (debug)
# =========================================================
@router.get("/corebank_data")
def get_corebank_data():
    return _get_corebank_table()
