# intention/action_service.py
# ===============================================
# 🧩 Action/Core Functions Layer (LOCKED)
# -----------------------------------------------
# - domain_model.json mapping (reply_text/pdf/static_voice_file/display_mn)
# - static/tts wav/mp3 resolver
# - corebank_reply.csv dynamic reload bridge
# - secure reply builder (balance, rates, etc)
# NO CLARIFY LOGIC (removed)
# ===============================================

import os
import csv
import json
from typing import Dict, Any, Optional


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENT_DIR = os.path.join(BASE_DIR, "intention")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TTS_DIR = os.path.join(STATIC_DIR, "tts")

DM_PATH = os.path.join(INTENT_DIR, "domain_model.json")
COREBANK_CSV = os.path.join(STATIC_DIR, "corebank_reply.csv")

os.makedirs(TTS_DIR, exist_ok=True)

_last_load_time = 0.0
_cached_corebank: Dict[str, str] = {}


def _read_corebank_csv(path: str) -> Dict[str, str]:
    data = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "name" in row and "text" in row:
                    k = row["name"].strip().lower()
                    v = row["text"].strip()
                    data[k] = v
    except Exception:
        return {}
    return data


def get_corebank_table(dlog=None) -> Dict[str, str]:
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
#

def attach_corebank_data(intent_key: str, dlog=None) -> Optional[Dict[str, str]]:
    table = get_corebank_table(dlog)
    keys = [k.lower() for k in INTENT_KEY_MAP.get(intent_key, [])]
    if not keys or not table:
        return None
    out = {k: table[k] for k in keys if k in table}
    if out:
        dlog and dlog(f"💾 CoreBank attach keys={list(out.keys())}")
    return out or None


def build_secure_display(intent: str, cb: Optional[Dict[str, str]]) -> str:
    if not cb:
        return ""

    if intent == "check_balance":
        lines = []
        if "account_type" in cb:
            lines.append(f"Таны дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb:
            lines.append(f"Үлдэгдэл : {cb['account_balance']}")
        if "customer_name" in cb:
            lines.append(f"Харилцагчийн нэр : {cb['customer_name']}")
        return "\n".join(lines)

    if intent == "branch_hours" and "branch_hours" in cb:
        return f"Манай салбаруудын ажлын цаг : {cb['branch_hours']}"

    if intent == "exchange_rate":
        parts = []
        if "usd_rate" in cb:
            parts.append(f"USD : {cb['usd_rate']}")
        if "eur_rate" in cb:
            parts.append(f"EUR : {cb['eur_rate']}")
        if "loan_rate" in cb:
            parts.append(f"Зээлийн хүү : {cb['loan_rate']}")
        return "  ".join(parts)

    if intent == "customer_info" and "customer_name" in cb:
        return f"Харилцагчийн нэр : {cb['customer_name']}"

    if intent == "contact_support" and "branch_phone" in cb:
        return f"☎️ Холбоо барих утас : {cb['branch_phone']}"

    if intent == "account_details":
        lines = []
        if "account_type" in cb:
            lines.append(f"Дансны төрөл : {cb['account_type']}")
        if "account_balance" in cb:
            lines.append(f"Үлдэгдэл : {cb['account_balance']}")
        return "\n".join(lines)

    return ""


def get_static_voice_path(static_file: str, static_dir: str = STATIC_DIR) -> Optional[str]:
    """
    Supports:
      - "tts/foo.wav"
      - "tts/foo.mp3"
    Prefers mp3 if exists, else wav.
    Returns path relative to /static, e.g. "tts/foo.mp3"
    """
    rel = (static_file or "").strip().lstrip("/")
    if not rel:
        return None

    # normalize: if caller provided without "tts/", still allow:
    if not rel.startswith("tts/"):
        rel = f"tts/{rel}"

    # Build candidate absolute paths
    base = rel.replace(".wav", "").replace(".mp3", "")
    mp3_rel = f"{base}.mp3"
    wav_rel = f"{base}.wav"

    mp3_abs = os.path.join(static_dir, mp3_rel)
    wav_abs = os.path.join(static_dir, wav_rel)

    if os.path.exists(mp3_abs):
        return mp3_rel
    if os.path.exists(wav_abs):
        return wav_rel
    return None


def load_domain_model(dlog=None) -> Dict[str, Any]:
    try:
        if os.path.exists(DM_PATH) and os.path.getsize(DM_PATH) > 2:
            with open(DM_PATH, "r", encoding="utf-8") as f:
                dm = json.load(f)
            if isinstance(dm, dict):
                return dm
        dlog and dlog(f"⚠️ domain_model missing/empty at {DM_PATH}")
    except Exception as e:
        dlog and dlog(f"⚠️ Failed to load domain_model.json: {e}")
    return {}


def get_domain_action(intent: str, dlog=None) -> Dict[str, Any]:
    dm = load_domain_model(dlog)
    action = dm.get(intent)

    if not isinstance(action, dict):
        return {
            "action": "voice_reply",
            "reply_text": "Уучлаарай, таны хүсэлтийг ойлгосонгүй.",
            "pdf_url": None,
            "voice_url": None,
            "static_voice_file": None,
            "force_static": False,
            "display_mn": None,
        }

    display_mn = action.get("display_mn")

    pdf_url = action.get("pdf_url")
    if isinstance(pdf_url, str) and pdf_url.strip():
        pdf_url = pdf_url.strip()
        if not pdf_url.startswith("/"):
            pdf_url = f"/static/{pdf_url.lstrip('/')}"
    else:
        pdf_url = None

    svf = (action.get("static_voice_file") or "").strip().lstrip("/")
    vurl = action.get("voice_url")

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
        "reply_text": action.get("reply_text", "") or "",
        "pdf_url": pdf_url,
        "voice_url": vurl,
        "force_static": action.get("force_static", False),
        "static_voice_file": action.get("static_voice_file"),
        "display_mn": display_mn,
    }


def apply_core_functions(
    pred_intent: str,
    conf: float,
    user_text: str,
    dlog=None,
) -> Dict[str, Any]:
    """
    NO-CLARIFY core functions.
    Returns fields merged into router response (frontend schema preserved).
    """
    domain_action = get_domain_action(pred_intent, dlog)

    cb = attach_corebank_data(pred_intent, dlog)
    secure_part = build_secure_display(pred_intent, cb).strip() if cb else ""

    base_reply = (domain_action.get("reply_text") or "").strip()
    reply_text = base_reply or "Таны хүсэлтийг хүлээн авлаа."
    if secure_part:
        reply_text = f"{reply_text}\n{secure_part}"

    return {
        "intent_override": None,
        "action": domain_action.get("action", "voice_reply"),
        "reply_text": reply_text,
        "pdf_url": domain_action.get("pdf_url"),
        "voice_url": domain_action.get("voice_url"),
        "corebank_data": cb,
        "intent_choices": None,
        "clarify_keyword": None,
        "display_mn": domain_action.get("display_mn"),
    }
