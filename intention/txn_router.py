from __future__ import annotations

import os, re, uuid, time, importlib
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import httpx
from httpx import ASGITransport

from intention.number_resolver import (
    resolver,
    resolve_amount_logic,
    resolve_account_logic,
    resolve_iban_logic,
    resolve_money
)

router = APIRouter()
TXN_APP_IMPORT = os.getenv("TXN_APP_IMPORT", "app:app")

_PREVIEW_CACHE: Dict[str, Dict[str, Any]] = {}

# ✅ NEW: Stream cache for chunking / append mode (IBAN + Account only)
_STREAM_CACHE: Dict[str, Dict[str, Any]] = {}

# ✅ FINAL SUCCESS REPLY (minimal principle)
SUCCESS_TXN_TEXT = "Таны гүйлгээ, төлбөр шилжүүлгийн хүсэлт амжилттай илгээгдлээ!"


class TxnPreviewIn(BaseModel):
    user_id: str
    iban_full: str = ""
    account_digits: str = ""
    amount_value: float
    memo: str = ""


def _mask(d: str) -> str:
    digits = re.sub(r"\D+", "", d or "")
    if len(digits) <= 4:
        return digits
    return ("•" * (len(digits) - 4)) + digits[-4:]


def _stream_key(user_id: str, slot: str, stream_id: str) -> str:
    sid = (stream_id or "").strip() or "default"
    return f"{user_id}::{slot}::{sid}"


def _stream_gc(now: float, ttl_sec: float = 180.0):
    # keep cache small & safe
    dead = []
    for k, v in _STREAM_CACHE.items():
        ts = float(v.get("ts", 0.0) or 0.0)
        if now - ts > ttl_sec:
            dead.append(k)
    for k in dead:
        _STREAM_CACHE.pop(k, None)


def _append_digits(prev: str, chunk: str) -> str:
    p = re.sub(r"\D+", "", prev or "")
    c = re.sub(r"\D+", "", chunk or "")
    return p + c


async def _transcribe_internal(file_bytes: bytes, filename: str, mime: str, uid: str) -> str:
    mod_name, obj_name = TXN_APP_IMPORT.split(":", 1)
    app_instance = getattr(importlib.import_module(mod_name), obj_name)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app_instance),
        base_url="http://app"
    ) as client:
        files = {"file": (filename, file_bytes, mime)}
        r = await client.post("/transcribe", files=files, data={"user_id": uid}, timeout=60.0)
        return r.json().get("user_text", "")


@router.post("/normalize_slot")
async def normalize_slot(
    slot: str = Form(...),
    user_id: str = Form("usr001"),
    file: UploadFile = File(...),

    # ✅ NEW (optional) — for chunking tactic test
    append: str = Form("0"),        # "1" => append digits to stream cache
    stream_id: str = Form(""),      # unique id per recording session
    reset_stream: str = Form("0"),  # "1" => clear cached digits first
):
    audio_data = await file.read()

    now = time.time()
    _stream_gc(now)

    # 1) STT
    raw_text = await _transcribe_internal(audio_data, file.filename, file.content_type, user_id)

    # ✅ clean_text MUST exist
    clean_text = raw_text or ""

    # base response
    res_data: Dict[str, Any] = {
        "slot": slot,
        "stt_text": raw_text or "EMPTY",
        "normalized": {},
        # harmless extra fields for debug
        "append": append,
        "stream_id": stream_id,
        "reset_stream": reset_stream,
    }

    # streaming only for iban/account in append mode
    is_stream = (append == "1") and (slot in ("iban", "account"))
    key = _stream_key(user_id, slot, stream_id)

    if reset_stream == "1":
        _STREAM_CACHE.pop(key, None)

    # 🟡 IBAN
    if slot == "iban":
        data, conf = resolve_iban_logic(clean_text)
        val_full = data.get("value", "")               # expected "MN...."
        val_digits = val_full.replace("MN", "")        # digits only

        if is_stream:
            prev_digits = (_STREAM_CACHE.get(key) or {}).get("digits", "")
            combined_digits = _append_digits(prev_digits, val_digits)
            _STREAM_CACHE[key] = {"digits": combined_digits, "ts": now}

            combined_full = f"MN{combined_digits}" if combined_digits else ""
            res_data["normalized"] = {
                "iban_full": combined_full,
                "iban_digits": combined_digits,
                "chunk_digits": val_digits,
                "combined_digits": combined_digits,
                "fixed_text": data.get("fixed_text", clean_text),
                "masked": _mask(combined_full),
                "confidence": conf,
            }
        else:
            res_data["normalized"] = {
                "iban_full": val_full,
                "iban_digits": val_digits,
                "fixed_text": data.get("fixed_text", clean_text),
                "masked": _mask(val_full),
                "confidence": conf
            }

    # 🔵 Account
    elif slot == "account":
        data, conf = resolve_account_logic(clean_text)
        chunk_digits = data.get("value", "")

        if is_stream:
            prev_digits = (_STREAM_CACHE.get(key) or {}).get("digits", "")
            combined_digits = _append_digits(prev_digits, chunk_digits)
            _STREAM_CACHE[key] = {"digits": combined_digits, "ts": now}

            res_data["normalized"] = {
                "account_digits": combined_digits,
                "chunk_digits": chunk_digits,
                "combined_digits": combined_digits,
                "fixed_text": data.get("fixed_text", clean_text),
                "masked": _mask(combined_digits),
                "confidence": conf
            }
        else:
            res_data["normalized"] = {
                "account_digits": chunk_digits,
                "fixed_text": data.get("fixed_text", clean_text),
                "masked": _mask(chunk_digits),
                "confidence": conf
            }

    # 🔴 Amount (UNCHANGED — no streaming in this test)
    elif "amount" in slot:
        if slot == "amount_mongo":
            data, conf = resolve_amount_logic(clean_text)
            val = min(99, data.get("value", 0))
            fixed = data.get("fixed_text", clean_text)

        elif slot == "amount_tugrug":
            data, conf = resolve_amount_logic(clean_text)
            val = data.get("value", 0)
            fixed = data.get("fixed_text", clean_text)

        else:
            tug, mon, conf = resolve_money(clean_text)
            val = float(tug + (mon / 100))
            _, fixed = resolver.resolve(clean_text, mode="amount")

        res_data["normalized"] = {
            "amount_value": val,
            "fixed_text": fixed or clean_text,
            "pretty": f"{val:,} ₮",
            "confidence": conf
        }

    return res_data


@router.post("/preview")
async def preview_txn(payload: TxnPreviewIn):
    pid = f"prev_{uuid.uuid4().hex[:8]}"
    fee = max(100, int(payload.amount_value * 0.002))
    target = payload.iban_full if payload.iban_full else payload.account_digits
    _PREVIEW_CACHE[pid] = {"payload": payload.dict(), "fee": fee, "ts": time.time()}
    return {
        "txn_preview_id": pid,
        "verified_account": len(re.sub(r"\D", "", target)) >= 7,
        "limit_ok": payload.amount_value <= 50_000_000,
        "to_account_masked": _mask(target),
        "amount_value": payload.amount_value,
        "fee": fee,
        "total_debit": payload.amount_value + fee
    }


@router.post("/execute")
async def execute_txn(payload: Dict[str, Any]):
    prev_id = payload.get("txn_preview_id")
    if not prev_id or prev_id not in _PREVIEW_CACHE:
        raise HTTPException(status_code=400, detail="Invalid session")

    # ✅ FINAL: success reply included
    return {
        "ok": True,
        "transfer_id": f"TXN-{uuid.uuid4().hex[:6].upper()}",
        "reply_text": SUCCESS_TXN_TEXT,
    }
