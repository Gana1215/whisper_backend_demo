# ===============================================
# 💬 Banking Intention Router
#    Phase 2 — v3.1.2 (Dynamic Clarify + Yesui TTS Cache + Direct-Intent Buttons)
# -----------------------------------------------
# ✅ Keeps v3.1.1 logic fully intact (dynamic clarify, Yesui TTS cache)
# ✅ Adds 1:1 direct-intent handling for clarify-button clicks
# ✅ Clarify MP3 auto-cache via mn-MN-YesuiNeural (edge-tts)
# ✅ Fallback to /static/tts/fallback_clarify.mp3 on failure
#
# ✅ SURGICAL PATCH ONLY (NO OTHER LOGIC TOUCHED):
# 1) domain_model loaded dynamically (prevents empty-startup poisoning)
# 2) get_domain_action supports "static_voice_file" + legacy "voice_url"
# 3) /list_intents comes ONLY from domain_model.json keys (no hard-coded fallback)
#
# ✅ NEW SURGICAL PATCH (NO OTHER LOGIC TOUCHED):
# 4) /voice_intent returns timing fields + user_text for demo (stt_ms/processing_ms/total_ms)
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

# =========================================================
# ✅ SURGICAL PATCH #1: Dynamic domain_model loader
# =========================================================
def load_domain_model(dlog=None) -> Dict[str, Any]:
    try:
        if os.path.exists(DM_PATH) and os.path.getsize(DM_PATH) > 2:
            with open(DM_PATH, "r", encoding="utf-8") as f:
                dm = json.load(f)
            if isinstance(dm, dict):
                return dm
        dlog and dlog(f"⚠️ domain_model missing/empty at {DM_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load domain_model.json: {e}")
    return {}

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

    # ✅ PATCH #4: demo timing fields for backend mode panel
    user_text: Optional[str] = None
    stt_ms: Optional[int] = None
    processing_ms: Optional[int] = None
    total_ms: Optional[int] = None

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
# 🔹 Clarify decision (with direct-intent guard)
# =========================================================
def maybe_clarify_flow(text: str, pred_intent: str, conf: float, dlog) -> Optional[Dict[str, Any]]:
    text_norm = normalize_text(text)

    def strong_and_in_bucket(bucket: str) -> bool:
        related = collect_related_intents(bucket)
        return (conf >= 0.60) and (pred_intent in related)

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

# =========================================================
# ✅ SURGICAL PATCH #2: domain action resolver
# =========================================================
def get_domain_action(intent, dlog=None):
    dm = load_domain_model(dlog)
    action = dm.get(intent)

    if not isinstance(action, dict):
        return {"action": "voice_reply",
                "reply_text": "Уучлаарай, таны хүсэлтийг ойлгосонгүй.",
                "pdf_url": None,
                "voice_url": None}

    # pdf_url normalize (/static/...)
    pdf_url = action.get("pdf_url")
    if isinstance(pdf_url, str) and pdf_url.strip():
        pdf_url = pdf_url.strip()
        if not pdf_url.startswith("/"):
            pdf_url = f"/static/{pdf_url.lstrip('/')}"
    else:
        pdf_url = None

    # voice normalize
    svf = (action.get("static_voice_file") or "").strip().lstrip("/")
    vurl = action.get("voice_url")  # legacy

    if svf:
        resolved = get_static_voice_path(svf, STATIC_DIR)
        vurl = f"/static/{resolved}" if resolved else None
    elif isinstance(vurl, str) and vurl.strip():
        vurl = vurl.strip()
        if not vurl.startswith("/"):
            vurl = f"/static/{vurl.lstrip('/')}"
    else:
        vurl = None

    return {
        "action": action.get("action", "voice_reply"),
        "reply_text": action.get("reply_text", ""),
        "pdf_url": pdf_url,
        "voice_url": vurl,
        "force_static": action.get("force_static", False),
        "static_voice_file": action.get("static_voice_file"),
    }

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
        if "account_balance" in cb: lines.append(f"Üлдэгдэл : {cb['account_balance']}")
        return "\n".join(lines)
    return ""

# =========================================================
# 🔹 /classify (unchanged)
# =========================================================
@router.post("/classify", response_model=IntentResponse)
def classify_intent(text: str = Form(...)):
    dlog = mk_dlog(ENV_DIAG)
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

    domain_action = get_domain_action(pred_intent, dlog)
    result.update(domain_action)
    cb = attach_corebank_data(pred_intent, dlog)
    if cb: result["corebank_data"] = cb
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(pred_intent, cb).strip() if cb else ""
    result["reply_text"] = f"{base_reply}\n{secure_part}" if secure_part else base_reply or "Таны хүсэлтийг хүлээн авлаа."
    return IntentResponse(**result)

# =========================================================
# 🔹 /voice_intent (PATCHED: timings + user_text, NO other logic touched)
# =========================================================
@router.post("/voice_intent", response_model=IntentResponse)
async def classify_from_voice(user_id: str = Form(...), file: UploadFile = File(...)):
    dlog = mk_dlog(ENV_DIAG)

    t_total0 = time.perf_counter()

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

    # ✅ STT timing only
    t_stt0 = time.perf_counter()
    segs, _ = model.transcribe(path, language="mn", task="transcribe", vad_filter=True)
    text = "".join(s.text for s in segs).strip()
    t_stt1 = time.perf_counter()
    dlog(f"🎧 Transcribed → {text}")

    # ✅ Processing timing (intent + reply build)
    t_proc0 = time.perf_counter()

    result = classify_text(text, dlog)
    pred_intent, prob = result["intent"], result["confidence"]

    clarify = maybe_clarify_flow(text, pred_intent, prob, dlog)
    if clarify:
        t_proc1 = time.perf_counter()
        t_total1 = time.perf_counter()
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

            # ✅ timing payload
            user_text=text,
            stt_ms=int((t_stt1 - t_stt0) * 1000),
            processing_ms=int((t_proc1 - t_proc0) * 1000),
            total_ms=int((t_total1 - t_total0) * 1000),
        )

    domain_action = get_domain_action(pred_intent, dlog)
    result.update(domain_action)
    cb = attach_corebank_data(pred_intent, dlog)
    if cb: result["corebank_data"] = cb
    base_reply = domain_action.get("reply_text", "").strip()
    secure_part = build_secure_display(pred_intent, cb).strip() if cb else ""
    result["reply_text"] = f"{base_reply}\n{secure_part}" if secure_part else base_reply or "Таны хүсэлтийг хүлээн авлаа."

    t_proc1 = time.perf_counter()
    t_total1 = time.perf_counter()

    # ✅ attach timing fields to normal reply
    result["user_text"] = text
    result["stt_ms"] = int((t_stt1 - t_stt0) * 1000)
    result["processing_ms"] = int((t_proc1 - t_proc0) * 1000)
    result["total_ms"] = int((t_total1 - t_total0) * 1000)

    return IntentResponse(**result)

# =========================================================
# 🔹 /corebank_data (unchanged)
# =========================================================
@router.get("/corebank_data")
def get_corebank_data():
    return _get_corebank_table()

# =========================================================
# 🔹 /text_intent (patched for 1:1 button-intents)
# =========================================================
@router.post("/text_intent", response_model=IntentResponse)
async def classify_text_intent(payload: dict):
    dlog = mk_dlog(ENV_DIAG)
    intent_key = payload.get("intent")
    source = payload.get("source", "")
    text = payload.get("text", "")

    # 🟢 1:1 Direct intent from clarify button
    if intent_key and source == "clarify_click":
        dlog and dlog(f"👉 Direct 1:1 intent click → {intent_key}")
        domain_action = get_domain_action(intent_key, dlog)
        cb = attach_corebank_data(intent_key, dlog)
        base_reply = domain_action.get("reply_text", "").strip()
        secure_part = build_secure_display(intent_key, cb).strip() if cb else ""
        reply = f"{base_reply}\n{secure_part}" if secure_part else base_reply or "Таны хүсэлтийг хүлээн авлаа."
        return IntentResponse(
            intent=intent_key,
            confidence=1.0,
            entities={},
            action=domain_action.get("action", "voice_reply"),
            reply_text=reply,
            pdf_url=domain_action.get("pdf_url"),
            voice_url=domain_action.get("voice_url"),
            corebank_data=cb or {},
            base_intent=intent_key,
            base_confidence=1.0,
            fallback_used=False
        )

    # 🔹 Otherwise normal text classification
    if text:
        return classify_intent(text=text)

    # 🔸 Empty fallback
    return IntentResponse(
        intent="error",
        confidence=0.0,
        entities={},
        action="voice_reply",
        reply_text="⚠️ Хоосон хүсэлт ирсэн."
    )

# =========================================================
# ✅ SURGICAL PATCH #3: /list_intents ONLY from domain_model.json
# =========================================================
@router.get("/list_intents")
def list_intents():
    dlog = mk_dlog(True)  # force endpoint logs

    dm = load_domain_model(dlog)

    intents = []
    if isinstance(dm, dict):
        for key, cfg in dm.items():
            display = None
            if isinstance(cfg, dict):
                display = (
                    cfg.get("display_mn")
                    or cfg.get("display")
                    or cfg.get("title")
                    or cfg.get("display_name")
                )
            if not display:
                display = key.replace("_", " ").title()

            intents.append({"name": key, "display": display})

    intents.sort(key=lambda x: x["display"])

    payload = {"count": len(intents), "intents": intents}

    dlog("✅ /list_intents payload count:", payload["count"])
    dlog("✅ /list_intents intents array (first 50):")
    for i, it in enumerate(payload["intents"][:50]):
        dlog(f"   {i:02d}", it)

    try:
        dlog("✅ /list_intents payload JSON:", json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        dlog("⚠️ dump failed:", e)

    return payload
#This was a huge multi-day frontend + backend refactor:

# ✔ Dynamic clarify
# ✔ Dynamic domain_model
# ✔ Static MP3/WAV auto-resolver
# ✔ Timing overlay front/back
# ✔ WebGPU + WASM stable
# ✔ Clean UX + toggles
# ✔ Scrollable chat
# ✔ Production-ready hooks
# # ✔ Zero regressions