# ===============================================
# 📚 dataset_routes.py (v3.8 — Safe Text Update + Render Stable)
# ✅ Fixes text update crash (NoneType.strip)
# ✅ Handles both JSON + FormData for /dataset/update
# ✅ Fully mobile-safe decode for webm/mp4/m4a/aac
# ✅ Works with app.py v2.1
# ===============================================

import os, csv, io, datetime, shutil, tempfile, subprocess
from fastapi import APIRouter, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment

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

# ✅ Ensure CSV header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file_name", "text"])


# =====================================================
# ✳️ Helper: append metadata
# =====================================================
def append_metadata(file_name: str, text: str):
    rel_path = f"wavs/{os.path.basename(file_name)}"
    clean_text = text.strip() if text.strip() else "EMPTY_AUDIO"
    try:
        rows = []
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        if not any(r["file_name"] == rel_path for r in rows):
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([rel_path, clean_text])
            print(f"🧾 Added → {rel_path} | text='{clean_text}'")
        else:
            print(f"⚠️ Duplicate skipped: {rel_path}")
    except Exception as e:
        print(f"❌ append_metadata failed: {e}")


# =====================================================
# 🆕 /dataset/add — Save new sample
# =====================================================
@router.post("/add")
async def add_sample(file: UploadFile = UploadFile, text: str = Form(...)):
    """Add new audio file + text to dataset."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        clean_name = f"usr001_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        target_path = os.path.join(WAV_DIR, clean_name)

        # Detect format
        ct = (file.content_type or "").lower()
        ext = os.path.splitext(file.filename or "")[1].lower()
        fmt = "wav"
        if "webm" in ct or ext == ".webm":
            fmt = "webm"
        elif any(k in ct for k in ["mp4", "aac", "m4a"]) or ext in [".mp4", ".m4a", ".aac"]:
            fmt = "mp4"
        elif "ogg" in ct or ext == ".ogg":
            fmt = "ogg"

        tmp_path = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_path, format=fmt)
        except Exception as e:
            print(f"⚠️ pydub decode failed ({e}); trying ffmpeg fallback...")
            tmp_wav = tempfile.mktemp(suffix=".wav")
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            os.remove(tmp_wav)

        os.remove(tmp_path)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")
        append_metadata(clean_name, text)

        return {"status": "ok", "file_name": clean_name, "text": text}
    except Exception as e:
        print(f"❌ add_sample failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🔁 /dataset/update_audio — Re-record existing row
# =====================================================
@router.post("/update_audio")
async def update_audio(file: UploadFile, file_name: str = Form(...)):
    """Replace existing WAV (keeps metadata intact)."""
    try:
        clean_name = os.path.basename(file_name)
        target_path = os.path.join(WAV_DIR, clean_name)
        if not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail=f"{file_name} not found")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        ct = (file.content_type or "").lower()
        ext = os.path.splitext(file.filename or "")[1].lower()
        fmt = "wav"
        if "webm" in ct or ext == ".webm":
            fmt = "webm"
        elif any(k in ct for k in ["mp4", "aac", "m4a"]) or ext in [".mp4", ".m4a", ".aac"]:
            fmt = "mp4"
        elif "ogg" in ct or ext == ".ogg":
            fmt = "ogg"

        tmp_path = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_path, format=fmt)
        except Exception as e:
            print(f"⚠️ pydub decode failed ({e}); trying ffmpeg fallback...")
            tmp_wav = tempfile.mktemp(suffix=".wav")
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            os.remove(tmp_wav)

        os.remove(tmp_path)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")
        print(f"🔁 Re-record replaced → {target_path}")
        return {"status": "ok", "message": f"{clean_name} updated successfully"}
    except Exception as e:
        print(f"❌ update_audio failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 📝 /dataset/update — Update text only (Safe Version)
# =====================================================
@router.post("/update")
async def update_text(request: Request):
    """Update text field in metadata.csv for given file_name."""
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            file_name = data.get("file_name")
            text = data.get("text")
        else:
            form = await request.form()
            file_name = form.get("file_name")
            text = form.get("text")

        if not file_name:
            raise HTTPException(status_code=400, detail="file_name missing")

        # 🩹 Prevent NoneType crash — normalize text
        if text is None:
            text = ""
        clean_text = text.strip() if text.strip() else "EMPTY_AUDIO"

        clean_name = os.path.basename(file_name)
        print(f"✏️ Updating text for {clean_name} → '{clean_text}'")

        rows, updated = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["file_name"].endswith(clean_name):
                    r["text"] = clean_text
                    updated = True
                rows.append(r)

        if not updated:
            raise HTTPException(status_code=404, detail=f"{file_name} not found in metadata.csv")

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ Text updated for {clean_name}")
        return {"status": "ok", "message": f"{clean_name} text updated", "new_text": clean_text}
    except Exception as e:
        print(f"❌ update_text failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
