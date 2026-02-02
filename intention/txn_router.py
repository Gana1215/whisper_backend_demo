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
    file: UploadFile = File(...)
):
    audio_data = await file.read()

    # 1) Транскрипци хийх
    raw_text = await _transcribe_internal(audio_data, file.filename, file.content_type, user_id)

    # ✅ 2) clean_text MUST exist (bug fix)
    clean_text = raw_text or ""

    # 2b) Noise цэвэрлэх (optional — keep commented if you want)
    # clean_text = re.sub(r'\b(5173|8000|3000)\b', '', clean_text)

    # 3) Default бүтэц
    res_data = {
        "slot": slot,
        "stt_text": raw_text or "EMPTY",
        "normalized": {}
    }

    # 🟡 IBAN
    if slot == "iban":
        data, conf = resolve_iban_logic(clean_text)
        val = data.get("value", "")
        res_data["normalized"] = {
            "iban_full": val,
            "iban_digits": val.replace("MN", ""),
            "fixed_text": data.get("fixed_text", clean_text),
            "masked": _mask(val),
            "confidence": conf
        }

    # 🔵 Account
    elif slot == "account":
        data, conf = resolve_account_logic(clean_text)
        res_data["normalized"] = {
            "account_digits": data.get("value", ""),
            "fixed_text": data.get("fixed_text", clean_text),
            "masked": _mask(data.get("value", "")),
            "confidence": conf
        }

    # 🔴 Amount (Tugrug/Mongo)
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

    # ✅ FINAL: success reply included (works for both voice + text transaction UIs)
    return {
        "ok": True,
        "transfer_id": f"TXN-{uuid.uuid4().hex[:6].upper()}",
        "reply_text": SUCCESS_TXN_TEXT,
    }
