# ===============================================
# 🎙️ dataset_routes.py (v4.4 — Unified Export Edition)
# ✅ Based on v4.3 Stable Final (Locked)
# ✅ Phase 1 untouched (record_archive handling)
# ✅ Adds Phase 2 (intention_archive) to export ZIP
# ✅ Safe for Render + Local environments
# ===============================================

import os, csv, datetime, shutil, tempfile, subprocess, zipfile
from fastapi import APIRouter, UploadFile, Form, Request, HTTPException, File
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment, generators

router = APIRouter(prefix="/dataset", tags=["dataset"])

# =====================================================
# 🌐 Resolve dataset directory
# =====================================================
BASE_DIR = os.getcwd()
LOCAL_PERSISTENT = os.path.join(BASE_DIR, "local_persistent", "record_archive")
ENV_DATA_DIR = os.getenv("DATA_DIR")

if ENV_DATA_DIR and os.path.exists(ENV_DATA_DIR):
    ARCHIVE_DIR = ENV_DATA_DIR
elif os.path.isdir(LOCAL_PERSISTENT):
    ARCHIVE_DIR = LOCAL_PERSISTENT
else:
    ARCHIVE_DIR = os.path.join(BASE_DIR, "record_archive")

print(f"📦 dataset_routes using → {ARCHIVE_DIR}")

WAV_DIR = os.path.join(ARCHIVE_DIR, "wavs")
CSV_PATH = os.path.join(ARCHIVE_DIR, "metadata.csv")
os.makedirs(WAV_DIR, exist_ok=True)

# Ensure CSV header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file_name", "text"])


# =====================================================
# ✳️ Helpers
# =====================================================
def append_metadata(file_name: str, text: str):
    rel_path = f"wavs/{os.path.basename(file_name)}"
    clean_text = text.strip() if isinstance(text, str) and text.strip() else "EMPTY_AUDIO"
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    if not any((r.get("file_name") or "") == rel_path for r in rows):
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rel_path, clean_text])
        print(f"🧾 Added → {rel_path} | text='{clean_text}'")

def detect_format(content_type: str, filename: str):
    ct = (content_type or "").lower()
    ext = os.path.splitext(filename or "")[1].lower()
    if "webm" in ct or ext == ".webm":
        return "webm"
    if any(k in ct for k in ["mp4", "m4a", "aac"]) or ext in [".mp4", ".m4a", ".aac"]:
        return "mp4"
    if "ogg" in ct or ext == ".ogg":
        return "ogg"
    return "wav"

def normalize_rel_path(name: str) -> str:
    """Return 'wavs/<basename>' for any incoming path variant."""
    if not name:
        return ""
    return f"wavs/{os.path.basename(name)}"


# =====================================================
# ➕ /dataset/add_empty — Create placeholder record
# =====================================================
@router.post("/add_empty")
async def add_empty():
    """
    Add a new placeholder sample with a short beep and default text.
    WAV exported in 44.1 kHz mono (mobile-safe).
    """
    try:
        clean_name = f"usr001_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        target_path = os.path.join(WAV_DIR, clean_name)

        # 1s sine beep @ 440 Hz → 44.1kHz mono 16-bit
        beep = generators.Sine(440).to_audio_segment(duration=1000)
        beep = beep.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        beep.export(target_path, format="wav")

        placeholder_text = "Enter the text for voice recording"
        append_metadata(clean_name, placeholder_text)

        print(f"➕ Added empty record → {target_path}")
        return {"status": "ok", "file_name": clean_name, "text": placeholder_text}
    except Exception as e:
        print(f"❌ add_empty failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🧩 /dataset/add — Save real audio + text (Transcribe tab)
# =====================================================
@router.post("/add")
async def add_sample(file: UploadFile = File(...), text: str = Form(...)):
    """
    Save audio + text coming from the Transcribe tab.
    Decodes mobile/desktop formats and exports 44.1kHz mono WAV.
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        clean_name = f"usr001_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        target_path = os.path.join(WAV_DIR, clean_name)

        fmt = detect_format(file.content_type, file.filename)

        # Decode
        tmp_path = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_path, format=fmt)
        except Exception as e:
            print(f"⚠️ decode failed ({e}); ffmpeg fallback...")
            tmp_wav = tempfile.mktemp(suffix=".wav")
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            os.remove(tmp_wav)
        finally:
            os.remove(tmp_path)

        # Export canonical WAV
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")

        append_metadata(clean_name, text)
        print(f"💾 Saved from Transcribe → {target_path}")
        return {"status": "ok", "file_name": clean_name, "text": text}
    except Exception as e:
        print(f"❌ add_sample failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🔁 /dataset/update_audio — Overwrite WAV safely
# =====================================================
@router.post("/update_audio")
async def update_audio(file: UploadFile = File(...), file_name: str = Form(...)):
    """
    Overwrite an existing WAV by file name, keep existing text in CSV.
    If the row is missing in CSV, it appends a new row with a default text.
    """
    try:
        if not file_name:
            raise HTTPException(status_code=400, detail="file_name missing")

        clean_name = os.path.basename(file_name.replace("wavs/", ""))
        target_path = os.path.join(WAV_DIR, clean_name)
        rel_path = f"wavs/{clean_name}"

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        fmt = detect_format(file.content_type, file.filename)

        # Decode
        tmp_path = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_path, format=fmt)
        except Exception as e:
            print(f"⚠️ PyDub decode failed ({e}); using ffmpeg fallback...")
            tmp_wav = tempfile.mktemp(suffix=".wav")
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            os.remove(tmp_wav)
        finally:
            os.remove(tmp_path)

        # Export canonical WAV
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")
        print(f"🎙️ Re-recorded → {target_path}")

        # Ensure CSV row exists; keep text intact
        placeholder_text = "EMPTY_AUDIO"
        rows, found = [], False
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    fn = (r.get("file_name") or "").strip()
                    if fn.endswith(clean_name) or os.path.basename(fn) == clean_name:
                        found = True
                    rows.append(r)
        if not found:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([rel_path, placeholder_text])

        return {"status": "ok", "message": f"{clean_name} re-recorded successfully"}
    except Exception as e:
        print(f"❌ update_audio failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 📝 /dataset/update — Update text
# =====================================================
@router.post("/update")
async def update_text(request: Request):
    try:
        ctype = (request.headers.get("content-type") or "").lower()
        file_name = text = None

        if ctype.startswith("application/json"):
            data = await request.json()
            file_name = data.get("file_name") or data.get("filename") or data.get("fileName")
            text = data.get("text") or data.get("new_text")
        else:
            form = await request.form()
            file_name = form.get("file_name") or form.get("filename") or form.get("fileName")
            text = form.get("text") or form.get("new_text")

        if not file_name:
            raise HTTPException(status_code=400, detail="file_name missing")
        if text is None:
            raise HTTPException(status_code=400, detail="text missing")

        clean_name = os.path.basename(file_name.replace("wavs/", ""))
        print(f"✏️ Updating text for {clean_name} → '{text}'")

        rows, updated = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = (r.get("file_name") or "").strip()
                if fn.endswith(clean_name) or os.path.basename(fn) == clean_name:
                    r["text"] = (text or "").strip()
                    updated = True
                rows.append(r)

        if not updated:
            raise HTTPException(status_code=404, detail=f"{file_name} not found")

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)

        return {"status": "ok", "message": f"{clean_name} text updated", "new_text": text}
    except Exception as e:
        print(f"❌ update_text failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🗑️ /dataset/delete
# =====================================================
@router.post("/delete")
async def delete_sample(request: Request):
    try:
        if (request.headers.get("content-type") or "").lower().startswith("application/json"):
            data = await request.json()
            file_name = data.get("file_name")
        else:
            form = await request.form()
            file_name = form.get("file_name")

        if not file_name:
            raise HTTPException(status_code=400, detail="file_name missing")

        clean_name = os.path.basename(file_name)
        target_wav = os.path.join(WAV_DIR, clean_name)

        rows, found = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if os.path.basename(r.get("file_name", "")) == clean_name:
                    found = True
                    continue
                rows.append(r)

        if not found:
            return JSONResponse(status_code=404, content={"error": f"{file_name} not found"})

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)

        if os.path.exists(target_wav):
            os.remove(target_wav)
            print(f"🗑️ Deleted → {target_wav}")

        return {"status": "ok", "message": f"{clean_name} deleted"}
    except Exception as e:
        print(f"❌ delete_sample failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🧾 /dataset/list  ← NORMALIZES ALL ROWS
# =====================================================
@router.get("/list")
async def list_samples():
    try:
        if not os.path.exists(CSV_PATH):
            return {"count": 0, "samples": []}
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                normalized = {}
                for k, v in r.items():
                    if not k:
                        continue
                    key = k.strip()
                    val = v.strip() if isinstance(v, str) else v
                    if key == "file_name":
                        val = normalize_rel_path(val)
                    normalized[key] = val
                rows.append(normalized)
        return {"count": len(rows), "samples": rows}
    except Exception as e:
        print(f"❌ list_samples failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 📦 /dataset/export — Unified Phase 1 + Phase 2 Archive
# =====================================================
@router.get("/export")
async def export_dataset():
    """
    Export Phase 1 (record_archive) together with Phase 2 (intention_archive)
    into a single ZIP archive for unified dataset backup.
    """
    try:
        if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
            return JSONResponse(status_code=400, content={"error": "No dataset entries yet."})

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = tempfile.gettempdir()
        zip_path = os.path.join(tmp_dir, f"MongolianWhisper_FullDataset_{ts}.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # --- Phase 1: Speech Dataset ---
            for root, _, files in os.walk(ARCHIVE_DIR):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, os.path.dirname(ARCHIVE_DIR))
                    zipf.write(full, rel)

            # --- Phase 2: Intention Dataset ---
            BASE_DIR = os.getcwd()
            INTENT_ARCHIVE = os.path.join(BASE_DIR, "local_persistent", "intention_archive")
            meta2 = os.path.join(INTENT_ARCHIVE, "metadata.csv")
            wavs2 = os.path.join(INTENT_ARCHIVE, "wavs")

            if os.path.exists(meta2):
                zipf.write(meta2, "intention_archive/metadata.csv")
                print("📦 Added Phase 2 metadata.csv")

            if os.path.exists(wavs2):
                for f in os.listdir(wavs2):
                    if f.lower().endswith(".wav"):
                        zipf.write(os.path.join(wavs2, f), f"intention_archive/wavs/{f}")
                print("📦 Added Phase 2 WAV files")

        resp = FileResponse(
            zip_path,
            filename=f"MongolianWhisper_FullDataset_{ts}.zip",
            media_type="application/zip",
        )
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Archive-Generated-At"] = ts
        print(f"✅ Exported unified dataset → {zip_path}")
        return resp

    except Exception as e:
        print(f"❌ export_dataset failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
