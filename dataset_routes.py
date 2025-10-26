# ===============================================
# 🎙️ dataset_routes.py (v4.1 — Stable Update + Add Empty)
# ✅ Fix: text update supports both text/new_text keys
# ✅ Fix: re-record works for any row, keeps text intact
# ✅ New: /dataset/add_empty — creates placeholder "beep" record
# ✅ Placeholder text improved → "Enter the text for voice recording"
# ✅ No change to mobile/desktop audio format logic
# ===============================================

import os, csv, datetime, shutil, tempfile, subprocess
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

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file_name", "text"])


# =====================================================
# ✳️ Helper: append metadata
# =====================================================
def append_metadata(file_name: str, text: str):
    rel_path = f"wavs/{os.path.basename(file_name)}"
    clean_text = text.strip() if text.strip() else "EMPTY_AUDIO"
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    if not any(r["file_name"] == rel_path for r in rows):
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rel_path, clean_text])
        print(f"🧾 Added → {rel_path} | text='{clean_text}'")


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

        # Create 1-second sine beep at 440 Hz
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
# 🔁 /dataset/update_audio — Overwrite WAV safely
# =====================================================
@router.post("/update_audio")
async def update_audio(file: UploadFile = File(...), file_name: str = Form(...)):
    try:
        clean_name = os.path.basename(file_name.replace("wavs/", ""))
        target_path = os.path.join(WAV_DIR, clean_name)
        rel_path = f"wavs/{clean_name}"

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
            print(f"⚠️ decode failed ({e}); ffmpeg fallback...")
            tmp_wav = tempfile.mktemp(suffix=".wav")
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "44100", tmp_wav]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio = AudioSegment.from_file(tmp_wav, format="wav")
            os.remove(tmp_wav)
        finally:
            os.remove(tmp_path)

        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(target_path, format="wav")
        print(f"🎙️ Re-recorded → {target_path}")

        # Keep text intact or add default if missing
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
                csv.writer(f).writerow([rel_path, "EMPTY_AUDIO"])

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
                    r["text"] = text.strip()
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
        return FileResponse(zip_path, filename="MongolianWhisper_Dataset.zip", media_type="application/zip")
    except Exception as e:
        print(f"❌ export_dataset failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
