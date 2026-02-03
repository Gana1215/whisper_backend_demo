# intention/intent_router.py
# ===============================================
# 🌐 Intent Router (MINIMAL LOCKED — Single STT Source + Async Archive)
# -----------------------------------------------
# Endpoints:
#   POST /voice_intent (audio -> app.py /transcribe -> classify -> core)
#   POST /text_intent  (text or intent-key -> classify/bypass -> core)
#   GET  /list_intents (returns intents + display_mn from domain_model.json)
#
# ✅ app.py remains LOCKED (no changes required)
# ✅ ZERO duplicate STT logic for transcription (single source: /transcribe)
# ✅ Calls /transcribe IN-PROCESS via httpx.ASGITransport (no network hop)
# ✅ HARDENED: never leaks plain-text 500 (all crashes -> JSON {"detail": ...})
# ✅ Adds timings + user_text for demo panels
#
# ✅ Async persistence (does NOT affect timing panel)
# ✅ transfer_money patch: returns START PROMPT + transfer_start.mp3
#
# ✅ FIXED (THIS PATCH):
# - Restores GET /intent/list_intents so frontend buttons show again
# - Supports POST /intent/text_intent with {intent: "...", source:"direct_click"} (old behavior)
# - Keeps display_mn flowing consistently
# - ✅ BUGFIX: domain_model.json dict-shape supported (your current file)
# ===============================================

import os
import time
import csv
import json
import datetime
import tempfile
import subprocess
from typing import Dict, Any, Optional, List

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    BackgroundTasks,
)
from pydantic import BaseModel
import httpx

from intention.intent_hub import classify as hub_classify
from intention.action_service import apply_core_functions


ENV_DIAG = os.getenv("DIAG", "0") == "1"


def mk_dlog(enabled: bool):
    def _dlog(*a, **k):
        if enabled:
            print(*a, **k)
    return _dlog


router = APIRouter(tags=["Bank Intention Handler"])


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

    # kept for frontend compatibility
    intent_choices: Optional[List[str]] = None
    clarify_keyword: Optional[str] = None

    # display name from domain_model.json
    display_mn: Optional[str] = None

    # demo/timing fields
    user_text: Optional[str] = None
    stt_ms: Optional[int] = None
    processing_ms: Optional[int] = None
    total_ms: Optional[int] = None


# =====================================================
# ✅ Persistence paths (Render + Local safe)
# =====================================================
BASE_DIR = os.getcwd()
INTENTION_ARCHIVE_DIR = os.path.join(BASE_DIR, "local_persistent", "intention_archive")
INTENTION_WAV_DIR = os.path.join(INTENTION_ARCHIVE_DIR, "wavs")
INTENTION_META_CSV = os.path.join(INTENTION_ARCHIVE_DIR, "metadata.csv")

INTENTION_TEXT_DIR = os.path.join(INTENTION_ARCHIVE_DIR, "text")  # optional
os.makedirs(INTENTION_WAV_DIR, exist_ok=True)
os.makedirs(INTENTION_TEXT_DIR, exist_ok=True)

TEXT_ARCHIVE_MODE = os.getenv("TEXT_ARCHIVE_MODE", "csv").lower().strip()
# csv   -> append text_intent to metadata.csv only
# files -> write /text/*.txt + also append metadata.csv (file_name points to txt)


def _ensure_metadata_header(diag: bool = False):
    try:
        if os.path.exists(INTENTION_META_CSV) and os.stat(INTENTION_META_CSV).st_size > 10:
            return
        os.makedirs(INTENTION_ARCHIVE_DIR, exist_ok=True)
        with open(INTENTION_META_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "ts",
                    "user_id",
                    "kind",         # voice | text
                    "file_name",    # wavs/<...>.wav OR text/<...>.txt OR ""
                    "text",
                    "pred_intent",
                    "final_intent",
                    "confidence",
                    "base_intent",
                    "base_confidence",
                    "fallback_used",
                    "source",
                ]
            )
        if diag:
            print(f"🧾 [INTENTION_ARCHIVE] created header → {INTENTION_META_CSV}")
    except Exception as e:
        if diag:
            print(f"⚠️ [INTENTION_ARCHIVE] header ensure failed: {type(e).__name__}: {e}")


def _append_metadata_row(
    *,
    user_id: str,
    kind: str,
    file_name: str,
    text: str,
    pred_intent: str,
    final_intent: str,
    confidence: float,
    base_intent: Optional[str],
    base_confidence: float,
    fallback_used: bool,
    source: str,
    diag: bool = False,
):
    try:
        _ensure_metadata_header(diag=diag)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(INTENTION_META_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    ts,
                    user_id or "",
                    kind or "",
                    file_name or "",
                    (text or "").strip(),
                    pred_intent or "",
                    final_intent or "",
                    float(confidence or 0.0),
                    base_intent or "",
                    float(base_confidence or 0.0),
                    bool(fallback_used),
                    source or "",
                ]
            )
        if diag:
            print(f"🧾 [INTENTION_ARCHIVE] metadata append ({kind}) final={final_intent} file={file_name}")
    except Exception as e:
        if diag:
            print(f"⚠️ [INTENTION_ARCHIVE] metadata append failed: {type(e).__name__}: {e}")


def _write_text_file(user_id: str, text: str, base_name: str, diag: bool = False) -> str:
    try:
        os.makedirs(INTENTION_TEXT_DIR, exist_ok=True)
        abs_path = os.path.join(INTENTION_TEXT_DIR, f"{base_name}.txt")
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write((text or "").strip() + "\n")
        rel = f"text/{base_name}.txt"
        if diag:
            print(f"📝 [INTENTION_ARCHIVE] text saved → {abs_path}")
        return rel
    except Exception as e:
        if diag:
            print(f"⚠️ [INTENTION_ARCHIVE] text save failed: {type(e).__name__}: {e}")
        return ""


def _convert_and_save_wav_16k_mono_pcm16(
    *,
    raw_bytes: bytes,
    orig_filename: str,
    out_wav_name: str,
    diag: bool = False,
):
    if not raw_bytes:
        return

    tmp_in = None
    tmp_out = None
    try:
        ext = os.path.splitext(orig_filename or "")[1].lower() or ".bin"
        tmp_in = tempfile.mktemp(suffix=ext)
        tmp_out = tempfile.mktemp(suffix=".wav")

        with open(tmp_in, "wb") as f:
            f.write(raw_bytes)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_in,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                "-vn",
                "-f",
                "wav",
                tmp_out,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        os.makedirs(INTENTION_WAV_DIR, exist_ok=True)
        final_path = os.path.join(INTENTION_WAV_DIR, out_wav_name)
        with open(tmp_out, "rb") as r:
            wav_bytes = r.read()
        with open(final_path, "wb") as w:
            w.write(wav_bytes)

        if diag:
            print(f"💾 [INTENTION_ARCHIVE] saved 16k PCM16 WAV → {final_path} ({len(wav_bytes)} bytes)")

    except Exception as e:
        if diag:
            print(f"⚠️ [INTENTION_ARCHIVE] WAV convert/save failed: {type(e).__name__}: {e}")
    finally:
        for p in (tmp_in, tmp_out):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# =====================================================
# ✅ Domain model loader for /list_intents
# =====================================================
_DOMAIN_CACHE = {"path": None, "mtime": None, "intents": []}


def _find_domain_model_path() -> Optional[str]:
    """
    Find domain_model.json reliably on Render + Local.
    Supports env override: DOMAIN_MODEL_PATH
    """
    envp = os.getenv("DOMAIN_MODEL_PATH", "").strip()
    if envp and os.path.exists(envp):
        return envp

    # common places in this repo structure
    candidates = [
        os.path.join(BASE_DIR, "domain_model.json"),
        os.path.join(BASE_DIR, "intention", "domain_model.json"),
        os.path.join(BASE_DIR, "src", "intention", "domain_model.json"),
        os.path.join(BASE_DIR, "intents", "domain_model.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_intents_from_domain_model(diag: bool = False) -> List[Dict[str, str]]:
    """
    Returns: [{"name": "...", "display_mn": "..."}]

    Supports BOTH shapes:
      A) dict-of-intents (YOUR CURRENT domain_model.json):
         {"check_balance": {...}, "transfer_money": {...}}
      B) list-of-dicts (or wrapper dict with intents/items):
         {"intents":[{...}]}  OR  [{"name":"..."}]
    """
    try:
        path = _find_domain_model_path()
        if not path:
            if diag:
                print("⚠️ [list_intents] domain_model.json not found")
            return []

        mtime = os.path.getmtime(path)

        # cached
        if _DOMAIN_CACHE["path"] == path and _DOMAIN_CACHE["mtime"] == mtime and _DOMAIN_CACHE["intents"]:
            return _DOMAIN_CACHE["intents"]

        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        out: List[Dict[str, str]] = []

        # ✅ CASE A: dict-of-intents (your current file)
        if isinstance(j, dict) and j and all(isinstance(v, dict) for v in j.values()):
            for key, cfg in j.items():
                name = (key or "").strip()
                if not name:
                    continue
                display = (
                    cfg.get("display_mn")
                    or cfg.get("display")
                    or cfg.get("title")
                    or cfg.get("display_name")
                    or name
                )
                out.append({"name": name, "display_mn": str(display).strip()})

        # ✅ CASE B: wrapper dict with list inside
        elif isinstance(j, dict):
            raw = []
            if isinstance(j.get("intents"), list):
                raw = j["intents"]
            elif isinstance(j.get("items"), list):
                raw = j["items"]
            else:
                for v in j.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        raw = v
                        break

            for it in raw or []:
                if not isinstance(it, dict):
                    continue
                name = (it.get("name") or it.get("intent") or it.get("key") or "").strip()
                if not name:
                    continue
                display = (
                    it.get("display_mn")
                    or it.get("display")
                    or it.get("title")
                    or it.get("display_name")
                    or name
                )
                out.append({"name": name, "display_mn": str(display).strip()})

        # ✅ CASE C: top-level list
        elif isinstance(j, list):
            for it in j:
                if not isinstance(it, dict):
                    continue
                name = (it.get("name") or it.get("intent") or it.get("key") or "").strip()
                if not name:
                    continue
                display = (
                    it.get("display_mn")
                    or it.get("display")
                    or it.get("title")
                    or it.get("display_name")
                    or name
                )
                out.append({"name": name, "display_mn": str(display).strip()})

        _DOMAIN_CACHE["path"] = path
        _DOMAIN_CACHE["mtime"] = mtime
        _DOMAIN_CACHE["intents"] = out

        if diag:
            print(f"✅ [list_intents] loaded {len(out)} intents from {path}")

        return out

    except Exception as e:
        if diag:
            print(f"⚠️ [list_intents] load failed: {type(e).__name__}: {e}")
        return []


# =====================================================
# ✅ /list_intents (RESTORED for frontend)
# =====================================================
@router.get("/list_intents")
async def list_intents():
    intents = _load_intents_from_domain_model(diag=ENV_DIAG)
    return {"intents": intents}


# =====================================================
# ✅ transfer_money start patch
# =====================================================
TRANSFER_START_TEXT = "Та гүйлгээ хийх хүсэлт гаргажээ. Дараах мэдээллийг анхааралтай бөглөнө үү!"
TRANSFER_START_VOICE = "/static/tts/transfer_start.mp3"


def _apply_transfer_start_patch(final_intent: str, core: Dict[str, Any]) -> Dict[str, Any]:
    if final_intent != "transfer_money":
        return core

    core = dict(core or {})
    core["action"] = "start_txn_form"
    core["reply_text"] = TRANSFER_START_TEXT
    core["voice_url"] = TRANSFER_START_VOICE
    core["pdf_url"] = None
    return core


# =====================================================
# ✅ Single STT source: call LOCKED app.py /transcribe in-process
# =====================================================
async def _stt_via_locked_transcribe(request: Request, file: UploadFile) -> dict:
    b = await file.read()
    if not b:
        raise HTTPException(status_code=400, detail="Empty audio file")

    orig_b = b

    try:
        transport = httpx.ASGITransport(app=request.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://app") as c:
            r = await c.post(
                "/transcribe",
                files={
                    "file": (
                        file.filename or "audio.wav",
                        b,
                        file.content_type or "application/octet-stream",
                    )
                },
                timeout=120.0,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Internal /transcribe call failed: {type(e).__name__}: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"/transcribe failed: {r.text[:2000]}")

    try:
        j = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"/transcribe returned non-JSON: {type(e).__name__}: {e}")

    if not (j.get("user_text") or "").strip():
        raise HTTPException(status_code=400, detail="STT produced empty transcript")

    j["_raw_bytes"] = orig_b
    j["_orig_filename"] = file.filename or "audio.wav"
    return j


# =====================================================
# ✅ /voice_intent
# =====================================================
@router.post("/voice_intent", response_model=IntentResponse)
async def voice_intent(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    dlog = mk_dlog(ENV_DIAG)
    t_total0 = time.perf_counter()

    try:
        t_stt0 = time.perf_counter()
        stt_json = await _stt_via_locked_transcribe(request, file)
        t_stt1 = time.perf_counter()

        user_text = (stt_json.get("user_text") or "").strip()

        t_proc0 = time.perf_counter()

        brain = hub_classify(user_text, dlog) or {}
        pred_intent = brain.get("intent") or "unknown"
        conf = float(brain.get("confidence") or 0.0)

        core = (
            apply_core_functions(
                pred_intent=pred_intent,
                conf=conf,
                user_text=user_text,
                dlog=dlog,
            )
            or {}
        )

        final_intent = core.get("intent_override") or pred_intent
        core = _apply_transfer_start_patch(final_intent, core)

        t_proc1 = time.perf_counter()
        t_total1 = time.perf_counter()

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_wav_name = f"int{user_id}_{ts}.wav"
        raw_bytes = stt_json.get("_raw_bytes") or b""
        orig_name = stt_json.get("_orig_filename") or "audio.wav"

        background_tasks.add_task(
            _convert_and_save_wav_16k_mono_pcm16,
            raw_bytes=raw_bytes,
            orig_filename=orig_name,
            out_wav_name=out_wav_name,
            diag=ENV_DIAG,
        )

        background_tasks.add_task(
            _append_metadata_row,
            user_id=user_id,
            kind="voice",
            file_name=f"wavs/{out_wav_name}",
            text=user_text,
            pred_intent=pred_intent,
            final_intent=final_intent,
            confidence=conf,
            base_intent=brain.get("base_intent"),
            base_confidence=float(brain.get("base_confidence") or 0.0),
            fallback_used=bool(brain.get("fallback_used")),
            source="voice_intent",
            diag=ENV_DIAG,
        )

        return IntentResponse(
            intent=final_intent,
            confidence=conf,
            entities=brain.get("entities", {}),
            action=core.get("action", "voice_reply"),
            reply_text=core.get("reply_text"),
            pdf_url=core.get("pdf_url"),
            voice_url=core.get("voice_url"),
            corebank_data=core.get("corebank_data"),
            base_intent=brain.get("base_intent"),
            base_confidence=float(brain.get("base_confidence") or 0.0),
            fallback_used=bool(brain.get("fallback_used")),
            intent_choices=None,
            clarify_keyword=None,
            display_mn=core.get("display_mn"),
            user_text=user_text,
            stt_ms=int((t_stt1 - t_stt0) * 1000),
            processing_ms=int((t_proc1 - t_proc0) * 1000),
            total_ms=int((t_total1 - t_total0) * 1000),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"voice_intent crash: {type(e).__name__}: {e}")


# =====================================================
# ✅ /text_intent (supports BOTH free-text + direct intent-click)
# =====================================================
@router.post("/text_intent", response_model=IntentResponse)
async def text_intent(request: Request, background_tasks: BackgroundTasks, payload: dict):
    dlog = mk_dlog(ENV_DIAG)
    t_total0 = time.perf_counter()

    try:
        # ✅ accept BOTH
        #   {text: "..."}  (normal)
        #   {intent: "check_balance", source:"direct_click"} (button)
        text = (payload.get("text") or "").strip()
        direct_intent = (payload.get("intent") or payload.get("intent_key") or payload.get("name") or "").strip()

        if not text and not direct_intent:
            raise HTTPException(status_code=400, detail="text missing")

        user_id = (payload.get("user_id") or "usr001").strip()
        source = (payload.get("source") or ("direct_click" if direct_intent else "text_intent")).strip()

        t_proc0 = time.perf_counter()

        # -------------------------
        # 1) Button-click path: bypass classifier (confidence=1.0)
        # -------------------------
        if direct_intent:
            pred_intent = direct_intent
            conf = 1.0
            brain = {
                "intent": pred_intent,
                "confidence": conf,
                "entities": {},
                "base_intent": pred_intent,
                "base_confidence": conf,
                "fallback_used": False,
            }

            core = (
                apply_core_functions(
                    pred_intent=pred_intent,
                    conf=conf,
                    user_text=text or pred_intent,  # keep a non-empty user_text for logs/UI
                    dlog=dlog,
                )
                or {}
            )

        # -------------------------
        # 2) Normal free-text path: classify
        # -------------------------
        else:
            brain = hub_classify(text, dlog) or {}
            pred_intent = brain.get("intent") or "unknown"
            conf = float(brain.get("confidence") or 0.0)

            core = (
                apply_core_functions(
                    pred_intent=pred_intent,
                    conf=conf,
                    user_text=text,
                    dlog=dlog,
                )
                or {}
            )

        final_intent = core.get("intent_override") or pred_intent
        core = _apply_transfer_start_patch(final_intent, core)

        t_proc1 = time.perf_counter()
        t_total1 = time.perf_counter()

        def _persist_text():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rel_file = ""

            # store what the user actually typed, OR the intent key if it was a click
            persist_text = text or f"[intent_click] {direct_intent}"

            if TEXT_ARCHIVE_MODE == "files":
                base = f"txt{user_id}_{ts}"
                rel_file = _write_text_file(user_id, persist_text, base, diag=ENV_DIAG)

            _append_metadata_row(
                user_id=user_id,
                kind="text",
                file_name=rel_file,
                text=persist_text,
                pred_intent=pred_intent,
                final_intent=final_intent,
                confidence=conf,
                base_intent=brain.get("base_intent"),
                base_confidence=float(brain.get("base_confidence") or 0.0),
                fallback_used=bool(brain.get("fallback_used")),
                source=source,
                diag=ENV_DIAG,
            )

        background_tasks.add_task(_persist_text)

        # show user_text for demo/chat panels
        out_user_text = text or (payload.get("display") or payload.get("display_mn") or direct_intent)

        return IntentResponse(
            intent=final_intent,
            confidence=conf,
            entities=brain.get("entities", {}),
            action=core.get("action", "voice_reply"),
            reply_text=core.get("reply_text"),
            pdf_url=core.get("pdf_url"),
            voice_url=core.get("voice_url"),
            corebank_data=core.get("corebank_data"),
            base_intent=brain.get("base_intent"),
            base_confidence=float(brain.get("base_confidence") or 0.0),
            fallback_used=bool(brain.get("fallback_used")),
            intent_choices=None,
            clarify_keyword=None,
            display_mn=core.get("display_mn"),
            user_text=out_user_text,
            stt_ms=0,
            processing_ms=int((t_proc1 - t_proc0) * 1000),
            total_ms=int((t_total1 - t_total0) * 1000),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"text_intent crash: {type(e).__name__}: {e}")
