# ===============================================
# 📚 dataset_routes.py (v3.3 — DeleteFix + ReplaceFix + PCM Stable)
# ✅ Fixes delete mismatch (prefix handling)
# ✅ Ensures re-record replaces existing WAV correctly
# ✅ All other routes unchanged
# ===============================================

import os, csv, io, datetime, shutil, tempfile
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
    try:
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["file_name", "text"])

        existing = set()
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.add(r["file_name"])

        rel_path = f"wavs/{os.path.basename(file_name)}"
        clean_text = text.strip() if text.strip() else "EMPTY_AUDIO"

        if rel_path not in existing:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([rel_path, clean_text])
            print(f"🧾 Added to metadata.csv → {rel_path} | text='{clean_text}'")
        else:
            print(f"⚠️ Duplicate skipped: {rel_path}")

    except Exception as e:
        print(f"❌ Failed to append metadata: {e}")

# =====================================================
# 🔁 /dataset/update_audio — Re-record existing row
# =====================================================
@router.post("/update_audio")
async def update_audio(file: UploadFile, file_name: str = Form(...)):
    """Replace an existing WAV with new recording (keeps metadata intact)."""
    try:
        # Normalize path (strip wavs/ prefix if present)
        clean_name = os.path.basename(file_name)
        target_path = os.path.join(WAV_DIR, clean_name)

        if not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail=f"{file_name} not found")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        # --- Decode ---
        ct = file.content_type or ""
        fmt = None
        if "mp4" in ct or "aac" in ct:
            fmt = "mp4"
        elif "webm" in ct:
            fmt = "webm"
        elif "ogg" in ct:
            fmt = "ogg"
        elif "wav" in ct:
            fmt = "wav"

        try:
            audio = AudioSegment.from_file(io.BytesIO(contents), format=fmt)
        except Exception as e:
            print(f"⚠️ pydub decode failed ({e}); trying fallback...")
            try:
                data, sr = sf.read(io.BytesIO(contents), dtype="float32", always_2d=False)
                raw = AudioSegment(
                    data.tobytes(),
                    frame_rate=sr,
                    sample_width=4,
                    channels=1 if len(getattr(data, 'shape', [])) == 1 else data.shape[1],
                )
                audio = raw
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Unsupported audio format ({e2})")

        # --- Convert & overwrite ---
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(
            target_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1"],
        )

        print(f"🔁 Re-record replaced → {target_path}")
        return {"status": "ok", "message": f"{clean_name} updated successfully"}

    except Exception as e:
        print(f"❌ Failed to update audio: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =====================================================
# 🗑️ /dataset/delete — Safe delete (prefix agnostic)
# =====================================================
@router.post("/delete")
async def delete_sample(request: Request):
    """Delete dataset entry and its WAV file (handles wavs/ prefix)."""
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

        # --- Rewrite CSV without that row ---
        rows, found = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_clean = os.path.basename(r["file_name"])
                if existing_clean == clean_name:
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
            print(f"🗑️ Deleted file → {target_wav}")

        return {"status": "ok", "message": f"{clean_name} deleted"}

    except Exception as e:
        print(f"❌ Delete failed: {e}")
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
        print(f"❌ /dataset/list failed: {e}")
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

        return FileResponse(zip_path,
                            filename="MongolianWhisper_Dataset.zip",
                            media_type="application/zip")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
