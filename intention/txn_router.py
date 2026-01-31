# intention/txn_router.py
# ===============================================
# 🏦 Transaction Router (FINAL — Golden + MN number normalization)
# -----------------------------------------------
# Endpoints:
#   POST /txn/normalize_slot   (multipart: slot + audio -> digits/amount)
#   POST /txn/preview          (json: verify + fee + limit)
#   POST /txn/execute          (json: confirm -> returns receipt links)
#
# ✅ Does NOT modify intent_router.py
# ✅ Uses SINGLE STT SOURCE: /transcribe via ASGITransport (no network hop)
# ✅ MN "magic" number normalization:
#    - fuzzy: "зун"->"зуун", "дөрөг"->"дөрөв"
#    - digits: "нэг хоёр гурав" -> "123"
#    - amount: "Нэг сая ... мянга ... зуун ..." -> int
#
# ✅ Small additions:
#    - safe app import via TXN_APP_IMPORT="app:app"
#    - preview cache TTL + cleanup
# ===============================================

from __future__ import annotations

import os
import re
import uuid
import time
import importlib
from typing import Optional, Dict, Any, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

import httpx
from httpx import ASGITransport

from intention.mn_number_parser import (
    mn_words_to_digits,
    mn_words_to_amount,
    format_money_2dp,
)

router = APIRouter()

MN_PREFIX = "MN"

# Router config (safe defaults)
TXN_APP_IMPORT = os.getenv("TXN_APP_IMPORT", "app:app")  # e.g. "main:app"
TXN_PREVIEW_TTL_SEC = int(os.getenv("TXN_PREVIEW_TTL_SEC", "900"))  # 15 minutes

# ---------------------------
# helpers
# ---------------------------

def _json_error(msg: str, code: int = 400) -> HTTPException:
    return HTTPException(status_code=code, detail=msg)

def _only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def _mask_digits(d: str, show: int = 4) -> str:
    d = _only_digits(d)
    if not d:
        return ""
    if len(d) <= show:
        return d
    return ("•" * (len(d) - show)) + d[-show:]

def _parse_amount_digits_first(text: str) -> Optional[int]:
    """
    Fast path: if text contains digits, prefer digits-based parsing.
    Example: "1 200 000 төгрөг" -> 1200000
    """
    if not text:
        return None
    cand = _only_digits(text)
    if not cand:
        return None
    try:
        v = int(cand)
        return v if v > 0 else None
    except:
        return None

def _pretty_mnt(v: Optional[int]) -> str:
    if not isinstance(v, int):
        return ""
    return f"{v:,} ₮"

def _load_fastapi_app() -> Any:
    """
    Safe lazy loader for the FastAPI `app` instance.
    Configure with env: TXN_APP_IMPORT="app:app" or "main:app".
    """
    try:
        mod, obj = TXN_APP_IMPORT.split(":", 1)
        m = importlib.import_module(mod)
        app_obj = getattr(m, obj)
        return app_obj
    except Exception as e:
        raise _json_error(f"TXN_APP_IMPORT load failed ({TXN_APP_IMPORT}): {e}", 500)

async def _call_transcribe_in_process(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Calls the already-locked /transcribe endpoint IN-PROCESS using ASGITransport.
    Ensures we never duplicate STT logic.
    """
    fastapi_app = _load_fastapi_app()
    transport = ASGITransport(app=fastapi_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
        files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
        data = {"user_id": user_id}
        r = await client.post("/transcribe", files=files, data=data, timeout=120.0)
        if r.status_code != 200:
            raise _json_error(f"/transcribe failed: {r.text}", 500)
        return r.json()

def _now() -> float:
    return time.time()

# ---------------------------
# request/response models
# ---------------------------

class TxnPreviewIn(BaseModel):
    user_id: str = "usr001"
    iban_full: str = ""         # "MN" + digits
    account_digits: str = ""    # digits-only
    amount_value: int
    memo: str = ""


class TxnPreviewOut(BaseModel):
    txn_preview_id: str
    verified_account: bool
    limit_ok: bool
    to_account_masked: str
    amount_value: int
    fee: int
    total_debit: int


class TxnExecuteIn(BaseModel):
    user_id: str = "usr001"
    txn_preview_id: str
    confirm: bool = True


# ---------------------------
# in-memory preview cache (Render demo-safe)
# ---------------------------
_PREVIEW_CACHE: Dict[str, Dict[str, Any]] = {}

def _cleanup_preview_cache() -> None:
    """
    TTL cleanup to avoid memory growth on long-running instances.
    """
    if not _PREVIEW_CACHE:
        return
    ttl = max(60, int(TXN_PREVIEW_TTL_SEC))
    cutoff = _now() - ttl
    kill = [k for k, v in _PREVIEW_CACHE.items() if float(v.get("ts", 0)) < cutoff]
    for k in kill:
        _PREVIEW_CACHE.pop(k, None)


def _fee_rule(amount_value: int) -> int:
    """
    Demo fee rule. Replace with real corebank fee logic later.
    """
    fee = int(round(amount_value * 0.002))  # 0.2%
    return max(100, min(5000, fee))


def _limit_rule(amount_value: int) -> bool:
    """
    Demo daily limit rule. Replace later.
    """
    return amount_value <= 5_000_000


async def _verify_account_exists(iban_full: str, account_digits: str) -> bool:
    """
    Hook point for real corebank verification.
    For now: basic sanity checks.
    """
    if iban_full:
        d = _only_digits(iban_full.replace(MN_PREFIX, ""))
        return len(d) >= 10
    if account_digits:
        d = _only_digits(account_digits)
        return len(d) >= 8
    return False


def _dest_masked(iban_full: str, account_digits: str) -> str:
    if iban_full:
        d = _only_digits(iban_full.replace(MN_PREFIX, ""))
        return MN_PREFIX + _mask_digits(d, show=4)
    return _mask_digits(account_digits, show=4)


# ===========================
# 1) Normalize slot (multipart)
# ===========================
@router.post("/normalize_slot")
async def normalize_slot(
    slot: str = Form(...),                 # "iban" | "account" | "amount"
    user_id: str = Form("usr001"),
    prefix_locked: str = Form("MN"),       # always MN for IBAN
    current_text: str = Form(""),
    file: UploadFile = File(...),
):
    slot = (slot or "").strip().lower()
    if slot not in {"iban", "account", "amount"}:
        raise _json_error("slot must be one of: iban, account, amount", 400)

    b = await file.read()
    if not b:
        raise _json_error("empty upload", 400)

    stt = await _call_transcribe_in_process(
        file_bytes=b,
        filename=file.filename or f"slot_{slot}.webm",
        content_type=file.content_type or "application/octet-stream",
        user_id=user_id,
    )

    text = (stt.get("user_text") or "").strip()

    # ---- IBAN ----
    if slot == "iban":
        digits = _only_digits(text) or _only_digits(current_text)
        if not digits:
            digits = mn_words_to_digits(text) or mn_words_to_digits(current_text)

        masked = _mask_digits(digits, show=4)
        full = (prefix_locked or MN_PREFIX) + digits if digits else ""

        return {
            "slot": "iban",
            "stt_text": text,
            "normalized": {
                "iban_digits": digits,
                "iban_full": full,
                "masked": f"{prefix_locked or MN_PREFIX}{masked}" if digits else "",
            },
        }

    # ---- Account ----
    if slot == "account":
        digits = _only_digits(text) or _only_digits(current_text)
        if not digits:
            digits = mn_words_to_digits(text) or mn_words_to_digits(current_text)

        masked = _mask_digits(digits, show=4)
        return {
            "slot": "account",
            "stt_text": text,
            "normalized": {
                "account_digits": digits,
                "masked": masked,
            },
        }

    # ---- Amount ----
    amt = _parse_amount_digits_first(text)
    if amt is None:
        amt = mn_words_to_amount(text)
    if amt is None:
        amt = _parse_amount_digits_first(current_text)
    if amt is None:
        amt = mn_words_to_amount(current_text)

    return {
        "slot": "amount",
        "stt_text": text,
        "normalized": {
            "amount_value": amt,
            "pretty": _pretty_mnt(amt) if amt else "",
            "money_2dp": format_money_2dp(amt) if amt else "",
        },
    }


# ===========================
# 2) Preview (json)
# ===========================
@router.post("/preview", response_model=TxnPreviewOut)
async def preview_txn(payload: TxnPreviewIn):
    _cleanup_preview_cache()

    iban_full = (payload.iban_full or "").strip().upper()
    account_digits = _only_digits(payload.account_digits)

    if iban_full:
        if not iban_full.startswith(MN_PREFIX):
            raise _json_error("iban_full must start with MN", 400)
        # normalize: allow "MN 12 34" etc
        iban_digits = _only_digits(iban_full.replace(MN_PREFIX, ""))
        iban_full = MN_PREFIX + iban_digits

    if not iban_full and not account_digits:
        raise _json_error("Missing destination: iban_full or account_digits", 400)

    if not isinstance(payload.amount_value, int) or payload.amount_value <= 0:
        raise _json_error("amount_value must be a positive integer", 400)

    verified = await _verify_account_exists(iban_full, account_digits)
    limit_ok = _limit_rule(payload.amount_value)

    fee = _fee_rule(payload.amount_value)
    total = payload.amount_value + fee

    dest_masked = _dest_masked(iban_full, account_digits)

    preview_id = f"txnprev_{uuid.uuid4().hex}"
    _PREVIEW_CACHE[preview_id] = {
        "ts": _now(),
        "user_id": payload.user_id,
        "iban_full": iban_full,
        "account_digits": account_digits,
        "amount_value": payload.amount_value,
        "fee": fee,
        "total_debit": total,
        "limit_ok": limit_ok,
        "verified_account": verified,
        "memo": (payload.memo or "").strip(),
        "to_account_masked": dest_masked,
    }

    return TxnPreviewOut(
        txn_preview_id=preview_id,
        verified_account=verified,
        limit_ok=limit_ok,
        to_account_masked=dest_masked,
        amount_value=payload.amount_value,
        fee=fee,
        total_debit=total,
    )


# ===========================
# 3) Execute (json)
# ===========================
@router.post("/execute")
async def execute_txn(payload: TxnExecuteIn):
    _cleanup_preview_cache()

    if not payload.confirm:
        raise _json_error("confirm=false (cancelled)", 400)

    item = _PREVIEW_CACHE.get(payload.txn_preview_id)
    if not item:
        raise _json_error("invalid txn_preview_id (expired or unknown)", 400)

    if item.get("user_id") != payload.user_id:
        raise _json_error("user_id mismatch", 403)

    if not item.get("verified_account"):
        raise _json_error("account verification failed", 400)

    if not item.get("limit_ok"):
        raise _json_error("limit check failed", 400)

    # Demo execute: return keys compatible with your existing reply system.
    # Later: replace with real corebank transfer call + real receipt.
    return {
        "ok": True,
        "transfer_id": f"tx_{uuid.uuid4().hex[:12]}",
        "to_account_masked": item.get("to_account_masked"),
        "amount_value": item.get("amount_value"),
        "amount_pretty": _pretty_mnt(item.get("amount_value")),
        "amount_money_2dp": format_money_2dp(item.get("amount_value")),
        "fee": item.get("fee"),
        "total_debit": item.get("total_debit"),
        "memo": item.get("memo"),
        # demo assets
        "pdf_url": "/static/docs/transfer_receipt.pdf",
        "voice_url": "/static/tts/transfer_success.mp3",
    }
