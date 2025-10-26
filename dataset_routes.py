# =====================================================
# 🟢 /dataset/add — Persistent Save (after /transcribe)
# =====================================================
"""
🚀 Purpose:
    This route is called when the user clicks the "💾 Save to DB" button.

📍 Flow Summary:
    1. User records audio in browser → browser sends blob to /transcribe
       - /transcribe decodes & transcribes it temporarily (no permanent save)
       - The browser can immediately playback its local blob
    2. When user clicks "Save to DB":
       - The frontend re-uploads that same recorded blob to this route (/dataset/add)
       - This route converts and stores a *permanent* version:
            ✅ Converts any format (AAC, MP4, WebM, WAV, etc.)
            ✅ Re-encodes to standard RIFF PCM WAV (mono, 44.1kHz, 16-bit)
            ✅ Saves it under /record_archive/wavs/
            ✅ Appends the filename + transcription text to metadata.csv

💡 Why this split:
    - /transcribe should remain fast and lightweight (no disk writes)
    - /dataset/add handles dataset persistence and uniform audio formatting
"""


# ===============================================
# 📚 dataset_routes.py (v3.0 — True PCM Export + Stable Save)
# ✅ Converts uploaded files to RIFF PCM WAV (mono, 44.1kHz, 16-bit)
# ✅ Prevents duplicate entries in metadata.csv
# ✅ Logs "EMPTY_AUDIO" when text is empty
# ✅ Compatible with Render persistent disk (/local_persistent/record_archive)
# ===============================================
# ===============================================
# 📚 dataset_routes.py (v3.1 — Stable PCM Export + List Fix)
# ===============================================

import os, csv, aiofiles, io, datetime, shutil, tempfile
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
# 🟢 /dataset/add — Converts & saves as PCM WAV
# =====================================================
@router.post("/add")
async def add_sample(file: UploadFile, text: str = Form(...)):
    try:
        # --- Sanitize & fix filename ---
        base = os.path.splitext(os.path.basename(file.filename))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base.startswith("usr001_") and len(base.split("_")) >= 3:
            final_name = f"{base}.wav"        # already has timestamp
        else:
            final_name = f"{base}_{timestamp}.wav"
        final_path = os.path.join(WAV_DIR, final_name)

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        # --- Detect format ---
        ct = file.content_type or ""
        fmt = None
        if any(k in ct for k in ["mp4", "aac", "m4a"]):
            fmt = "mp4"
        elif "webm" in ct:
            fmt = "webm"
        elif "ogg" in ct:
            fmt = "ogg"
        elif "wav" in ct:
            fmt = "wav"
        else:
            fmt = "wav"  # fallback

        # --- Decode (pydub + fallback) ---
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

        # --- Convert to standard PCM WAV ---
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(
            final_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1"],
        )
        print(f"💾 Saved PCM WAV → {final_path}")

        # --- Append metadata ---
        append_metadata(final_name, text)

        return {"status": "ok", "file_name": final_name, "text": text.strip()}

    except Exception as e:
        print(f"❌ Failed to add sample: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =====================================================
# 🧾 /dataset/list — Fixed version
# =====================================================
@router.get("/list")
async def list_samples():
    try:
        print(f"DEBUG: Listing from {CSV_PATH}")
        if not os.path.exists(CSV_PATH):
            return {"count": 0, "samples": []}

        rows = []
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
                rows.append(clean)

        print(f"DEBUG: Loaded {len(rows)} rows")
        return {"count": len(rows), "samples": rows}

    except Exception as e:
        print(f"❌ /dataset/list failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =====================================================
# 🧹 /cleanup and /export remain same
# =====================================================
@router.post("/cleanup")
async def cleanup_orphans():
    try:
        valid = set()
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    valid.add(os.path.basename(r["file_name"]))
        removed = 0
        for f in os.listdir(WAV_DIR):
            if f.endswith(".wav") and f not in valid:
                os.remove(os.path.join(WAV_DIR, f))
                removed += 1
        return {"status": "ok", "removed": removed}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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
        return FileResponse(zip_path, filename="dataset_export.zip", media_type="application/zip")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
