# ===============================================
# 💬 Banking Intention Router
#    Phase 2 — v3.3 (SmartDirect + Dynamic Clarify + Yesui TTS Cache)
# -----------------------------------------------
# ✅ Clarify-button clicks → 1:1 direct intent (skip classifier/confidence)
# ✅ Clear text fast-path (keywords) → direct intent (skip classifier)
# ✅ Generic/ambiguous → dynamic clarify (Mongolian buckets)
# ✅ Clarify MP3 auto-cache via mn-MN-YesuiNeural (edge-tts)
# ✅ Fallback to /static/tts/fallback_clarify.mp3 on failure
# ===============================================

import os, csv, json, time, re, subprocess, shlex
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from fastapi import APIRouter, UploadFile, File, Form
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
YESUI_VOICE = "mn-MN-YesuiNeural"  # 🔒 our helper

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
# 🔹 Mongolian keyword buckets (semantic grouping)
# =========================================================
INTENT_KEYWORDS: Dict[str, str] = {
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

INTENT_TITLES: Dict[str, str] = {
    "contact_support": "Харилцагчийн үйлчилгээ",
    "customer_info": "Мэдээлэл шинэчлэх үйлчилгээ",
    "open_account": "Данс нээх үйлчилгээ",
    "check_balance": "Үлдэгдэл шалгах",
    "account_details": "Дансны мэдээлэл",
    "show_transactions": "Гүйлгээний хуулга",
    "transfer_money": "Мөнгө шилжүүлэх",
    "savings_info": "Хадгаламжийн мэдээлэл",
    "loan_info": "Зээлийн мэдээлэл",
    "exchange_rate": "Валютын ханш",
    "branch_location": "Салбарын байршил",
    "branch_hours": "Ажлын цаг",
    "lost_card": "Карт хаалгах / тусламж",
}

# ✅ Fast-path intent keywords (no ML when matched)
FAST_PATH_KEYWORDS: Dict[str, List[str]] = {
    "open_account": ["данс нээх", "шинэ данс", "данс үүсгэх"],
    "check_balance": ["үлдэгдэл шалгах", "дансны үлдэгдэл"],
    "account_details": ["дансны мэдээлэл", "дансны дэлгэрэнгүй"],
    "show_transactions": ["хуулга", "гүйлгээний хуулга"],
    "transfer_money": ["мөнгө шилжүүлэх", "данс руу шилжүүлэх"],
    "exchange_rate": ["валютын ханш", "ханш"],
    "branch_hours": ["ажлын цаг", "цагийн хуваарь"],
    "branch_location": ["салбарын байршил", "салбар хаана"],
    "lost_card": ["карт хаах", "карт алдсан"],
    "loan_info": ["зээлийн мэдээлэл", "зээлийн хүү"],
    "savings_info": ["хадгаламжийн мэдээлэл"],
    "contact_support": ["харилцагчийн үйлчилгээ", "үйлчилгээний төв"],
    "customer_info": ["мэдээлэл шинэчлэх", "нэр солих"],
}

GENERIC_TRIGGERS = [
    "данс", "даанс", "данса", "дансны",
    "үйлчилгээ", "үйлчилгээнүүд", "ямар үйлчилгээ", "ямар ямар үйлчилгээ",
    "хуулга", "гүйлгээ", "хадгаламж", "зээл", "ханш"
]

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\u0400-\u04FFa-z0-9\s\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_token_match(text: str, keyword: str) -> bool:
    text = normalize_text(text)
    if keyword in text: 
        return True
    for tok in text.split():
        if tok == keyword:
            return True
        if len(tok) >= 3 and len(keyword) >= 3:
            if tok.startswith(keyword[:3]) or keyword.startswith(tok[:3]):
                return True
    return False

def collect_related_intents(word: str) -> List[str]:
    return [i for i, w in INTENT_KEYWORDS.items() if w == word]

def build_clarify_text(word: str, intents: List[str]) -> Tuple[str, List[str]]:
    titles = [INTENT_TITLES.get(i, i) for i in intents]
    header = f"Юуны үйлчилгээ вэ, тодруулна уу?\nМанай банкинд {word}анд хамаарах дараах үйлчилгээ байна:"
    bullet = "\n".join(f"- {t}" for t in titles)
    return f"{header}\n{bullet}", titles

def ensure_clarify_mp3(word: str, dlog) -> Optional[str]:
    safe = re.sub(r"[^a-zA-Z0-9_\u0400-\u04FF\-]", "_", word)
    out_rel = f"tts/clarify_{safe}.mp3"
    out_abs = os.path.join(STATIC_DIR, out_rel)
    if os.path.exists(out_abs):
        dlog and dlog(f"🎙️ Reusing cached clarify → {out_abs}")
        return f"/static/{out_rel}"
    tts_text = f"Юуны үйлчилгээ вэ, тодруулна уу? Манай банкинд {word}анд хамаарах дараах үйлчилгээ байна."
    try:
        cmd = f'python3 -m edge_tts --voice "{YESUI_VOICE}" --text {shlex.quote(tts_text)} --write-media {shlex.quote(out_abs)}'
        dlog and dlog(f"🛠️ Generating clarify MP3: {cmd}")
        subprocess.run(cmd, shell=True, check=True, timeout=25)
        if os.path.exists(out_abs):
            return f"/static/{out_rel}"
    except Exception as e:
        print(f"⚠️ edge-tts failed for '{word}': {e}")
    fb_rel = "tts/fallback_clarify.mp3"
    fb_abs = os.path.join(STATIC_DIR, fb_rel)
    if os.path.exists(fb_abs):
        return f"/static/{fb_rel}"
    return None

# =========================================================
# 🔹 Direct-intent resolution (used by buttons & fast-path)
# =========================================================
def resolve_direct_intent(intent_key: str, dlog) -> IntentResponse:
    # Use domain_model if present, else provide minimal default action
    domain_action = domain_model.get(intent_key, {"action": "voice_reply",
                                                  "reply_text": "Таны хүсэлтийг хүлээн авлаа."})
    result: Dict[str, Any] = {
        "intent": intent_key,
        "confidence": 1.0,                # direct selection = full confidence
        "entities": {},
        "action": domain_action.get("action", "voice_reply"),
        "pdf_url": domain_action.get("pdf_url"),
        "base_intent": intent_key,
        "base_confidence": 1.0,
        "fallback_used": False
    }

    # CoreBank (optional enrichment)
    cb = attach_corebank_data(intent_key, dlog)
    if cb: result["corebank_data"] = cb

    # Text body (+ secure display add-on if any)
    base_reply = (domain_action.get("reply_text") or "").strip()
    secure_part = build_secure_display(intent_key, cb).strip() if cb else ""
    result["reply_text"] = f"{base_reply}\n{secure_part}" if secure_part else (base_reply or "Таны хүсэлтийг хүлээн авлаа.")

    # Voice file resolution (prefer static if provided)
    svf = (domain_action.get("static_voice_file") or "").strip().lstrip("/")
    resolved = get_static_voice_path(svf, STATIC_DIR) if svf else None
    result["voice_url"] = f"/static/{resolved}" if resolved else None

    return IntentResponse(**result)

# =========================================================
# 🔹 Clarify decision (with direct-intent guard)
# =========================================================
def maybe_clarify_flow(text: str, pred_intent: str, conf: float, dlog) -> Optional[Dict[str, Any]]:
    """
    Trigger clarify when:
    - Text hits a generic trigger AND classifier isn't giving a strong, specific match in that bucket, OR
    - Confidence is low and bucket has multiple plausible intents.
    """
    text_norm = normalize_text(text)

    # Helper: if the predicted intent belongs to the bucket and is strong → don't clarify.
    def strong_and_in_bucket(bucket: str) -> bool:
        related = collect_related_intents(bucket)
        return (conf >= 0.60) and (pred_intent in related)

    # 1) Generic triggers → find bucket; skip clarify if strong specific match
    for kw in GENERIC_TRIGGERS:
        if fuzzy_token_match(text_norm, kw):
            bucket = "данс" if kw.startswith("даа") or kw.startswith("данс") else kw
            if strong_and_in_bucket(bucket):
                dlog and dlog("✅ Strong specific match inside bucket — no clarify.")
                return None
            related = collect_related_intents(bucket)
            if related:
                clarify_text, titles = build_clarify_text(bucket, related)
                vurl = ensure_clarify_mp3(bucket, dlog)
                return {
                    "action": "voice_reply",
                    "reply_text": clarify_text,
                    "voice_url": vurl,
                    "intent_choices": titles,
                    "clarify_keyword": bucket,
                }

    # 2) Low confidence → if bucket has multiple intents, clarify
    if conf < max(0.45, CONF_THRESH):
        bucket_word = INTENT_KEYWORDS.get(pred_intent)
        if bucket_word:
            related = collect_related_intents(bucket_word)
            if len(related) >= 2:
                clarify_text, titles = build_clarify_text(bucket_word, related)
                vurl = ensure_clarify_mp3(bucket_word, dlog)
                return {
                    "action": "voice_reply",
                    "reply_text": clarify_text,
                    "voice_url": vurl,
                    "intent_choices": titles,
                    "clarify_keyword": bucket_word,
                }
    return None

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
        if "account_type" in cb:    txt.append(f"Таны дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb: txt.append(f"Үлдэгдэл : {cb['account_balance']}")
        return "\n".join(txt)
    if intent == "branch_hours" and "branch_hours" in cb:
        return f"Манай салбаруудын ажлын өдрүүдэд ажиллах хуваарь : {cb['branch_hours']}"
    if intent == "exchange_rate":
        parts = []
        if "usd_rate" in cb:  parts.append(f"USD : {cb['usd_rate']}")
        if "eur_rate" in cb:  parts.append(f"EUR : {cb['eur_rate']}")
        if "loan_rate" in cb: parts.append(f"Зээлийн хүү : {cb['loan_rate']}")
        return "  ".join(parts)
    if intent == "customer_info" and "customer_name" in cb:
        return f"Харилцагчийн нэр : {cb['customer_name']}"
    if intent == "contact_support" and "branch_phone" in cb:
        return f"☎️ Холбоо барих утас : {cb['branch_phone']}"
    if intent == "account_details":
        lines = []
        if "account_type" in cb:    lines.append(f"Дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb: lines.append(f"Үлдэгдэл : {cb['account_balance']}")
        return "\n".join(lines)
    return ""

# =========================================================
# 🔹 /classify — with fast-path direct matching
# =========================================================
@router.post("/classify", response_model=IntentResponse)
def classify_intent(text: str = Form(...)):
    dlog = mk_dlog(ENV_DIAG)
    text_norm = normalize_text(text)

    # 0) 🔥 Fast-path: exact/clear keywords → direct intent (skip ML)
    for key, kws in FAST_PATH_KEYWORDS.items():
        if any(fuzzy_token_match(text_norm, kw) for kw in kws):
            dlog and dlog(f"⚡ Fast-path keyword match → {key}")
            return resolve_direct_intent(key, dlog)

    # 1) ML classifier
    result = classify_text(text, dlog)
    pred_intent, prob = result["intent"], result["confidence"]

    # 2) Clarify decision
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

    # 3) Direct reply for confident ML path
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

# =========================================================
# 🔹 /voice_intent — STT → classify (kept as-is)
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

    # Fast-path for voice text too
    text_norm = normalize_text(text)
    for key, kws in FAST_PATH_KEYWORDS.items():
        if any(fuzzy_token_match(text_norm, kw) for kw in kws):
            dlog and dlog(f"⚡ Fast-path (voice) keyword match → {key}")
            return resolve_direct_intent(key, dlog)

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

# =========================================================
# 🔹 /corebank_data (debug)
# =========================================================
@router.get("/corebank_data")
def get_corebank_data():
    return _get_corebank_table()

# =========================================================
# 🔹 /list_intents — Dynamic front-end button menu (kept)
# =========================================================
@router.get("/list_intents")
def list_intents():
    """
    Returns a minimal list of available intent names and display titles
    for front-end quick-access rounded buttons (no reply text).
    """
    intents = []

    # --- Prefer domain_model.json (Phase 2 standard)
    if os.path.exists(DM_PATH):
        try:
            with open(DM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in data.keys():
                intents.append({
                    "name": key,
                    "display": INTENT_TITLES.get(key) or key.replace("_", " ").title()
                })
        except Exception as e:
            print(f"⚠️ Failed to load domain_model.json: {e}")

    # --- Fallback to intents.csv (Phase 1 compatibility)
    elif os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row.get("intent", "")
                    if key:
                        intents.append({
                            "name": key,
                            "display": INTENT_TITLES.get(key) or key.replace("_", " ").title()
                        })
        except Exception as e:
            print(f"⚠️ Failed to read intents.csv: {e}")

    # ✅ Ensure unique + sorted output for clean UI
    seen = set()
    unique_intents = []
    for it in intents:
        if it["name"] not in seen:
            unique_intents.append(it)
            seen.add(it["name"])
    unique_intents.sort(key=lambda x: x["display"])

    return {"count": len(unique_intents), "intents": unique_intents}

# =========================================================
# 🔹 /text_intent — JSON payload:
#     { "text": "...", "intent": "open_account", "source": "clarify_click" }
# =========================================================
class TextIntentPayload(BaseModel):
    text: Optional[str] = None
    intent: Optional[str] = None
    source: Optional[str] = None

@router.post("/text_intent", response_model=IntentResponse)
async def classify_text_intent(payload: TextIntentPayload):
    dlog = mk_dlog(ENV_DIAG)

    # 1) If front-end sends a clarify button click → direct 1:1 intent
    if payload.intent and (payload.source == "clarify_click" or not payload.text):
        dlog and dlog(f"👉 Direct intent from UI: {payload.intent}")
        return resolve_direct_intent(payload.intent, dlog)

    # 2) If text present, try fast-path keywords → direct intent
    if payload.text:
        text_norm = normalize_text(payload.text)
        for key, kws in FAST_PATH_KEYWORDS.items():
            if any(fuzzy_token_match(text_norm, kw) for kw in kws):
                dlog and dlog(f"⚡ Fast-path (text_intent) keyword match → {key}")
                return resolve_direct_intent(key, dlog)

        # 3) Else, fallback to /classify logic (kept identical)
        return classify_intent(text=payload.text)

    # 4) Empty payload
    return IntentResponse(
        intent="error",
        confidence=0.0,
        entities={},
        action="voice_reply",
        reply_text="⚠️ Хоосон хүсэлт ирсэн."
    )
