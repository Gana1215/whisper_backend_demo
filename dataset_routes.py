# ===============================================
# 📚 dataset_routes.py (v3.7 — Render Stable + Mobile WebM Fix)
# ✅ Adds /dataset/add and /dataset/update (for DB + text edit)
# ✅ Fully mobile-safe decode for webm/mp4/m4a/aac
# ✅ Works with app.py v2.1
# ===============================================

import os, csv, io, datetime, shutil, tempfile, subprocess
from fastapi import APIRouter, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
import soundfile as sf

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

        # Temp decode
        tmp_path = tempfile.mktemp(suffix=f".{fmt}")
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)

        try:
            audio = AudioSegment.from_file(tmp_path, format=fmt)
        except Exception as e:
            print(f"⚠️ pydub decode failed ({e}); trying ffmpeg pipe fallback...")
            try:
                tmp_wav = tempfile.mktemp(suffix=".wav")
                cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                audio = AudioSegment.from_file(tmp_wav, format="wav")
                os.remove(tmp_wav)
            except Exception as e2:
                os.remove(tmp_path)
                raise HTTPException(status_code=400, detail=f"Unsupported audio format ({e2})")

        os.remove(tmp_path)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")
        append_metadata(clean_name, text)

        return {"status": "ok", "file_name": clean_name, "text": text}
    except Exception as e:
        print(f"❌ add_sample failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🔁 /dataset/update_audio — Re-record existing row (Mobile Safe)
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
            print(f"⚠️ pydub decode failed ({e}); trying ffmpeg pipe fallback...")
            try:
                tmp_wav = tempfile.mktemp(suffix=".wav")
                cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                audio = AudioSegment.from_file(tmp_wav, format="wav")
                os.remove(tmp_wav)
            except Exception as e2:
                os.remove(tmp_path)
                raise HTTPException(status_code=400, detail=f"Unsupported audio format ({e2})")

        os.remove(tmp_path)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")
        print(f"🔁 Re-record replaced → {target_path}")
        return {"status": "ok", "message": f"{clean_name} updated successfully"}

    except Exception as e:
        print(f"❌ update_audio failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 📝 /dataset/update — Update text only
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

        clean_name = os.path.basename(file_name)
        rows = []
        updated = False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if os.path.basename(r["file_name"]) == clean_name:
                    r["text"] = text.strip()
                    updated = True
                rows.append(r)

        if not updated:
            raise HTTPException(status_code=404, detail=f"{file_name} not found")

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)

        return {"status": "ok", "message": f"{clean_name} text updated"}
    except Exception as e:
        print(f"❌ update_text failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🗑️ /dataset/delete
# =====================================================
@router.post("/delete")
async def delete_sample(request: Request):
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
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
                if os.path.basename(r["file_name"]) == clean_name:
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
# 🧾 /dataset/list
# =====================================================
@router.get("/list")
async def list_samples():
    try:
        if not os.path.exists(CSV_PATH):
            return {"count": 0, "samples": []}
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                {k.strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in r.items() if k}
                for r in reader
            ]
        return {"count": len(rows), "samples": rows}
    except Exception as e:
        print(f"❌ list_samples failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 📦 /dataset/export
# =====================================================
@router.get("/export")
async def export_dataset():
    try:
        if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
            return JSONResponse(status_code=400, content={"error": "No dataset entries yet."})

        tmp_dir = tempfile.gettempdir()
        zip_base = os.path.join(tmp_dir, "dataset_export")
        zip_path = f"{zip_base}.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)

        shutil.make_archive(zip_base, "zip", ARCHIVE_DIR)
        print(f"📦 Dataset exported → {zip_path}")

        return FileResponse(
            zip_path,
            filename="MongolianWhisper_Dataset.zip",
            media_type="application/zip"
        )
    except Exception as e:
        print(f"❌ export_dataset failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
