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

import os, csv, aiofiles, io, datetime, shutil, tempfile
from fastapi import APIRouter, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
import soundfile as sf

router = APIRouter(prefix="/dataset", tags=["dataset"])

# =====================================================
# 🌐 Resolve shared persistent dataset directory
# =====================================================
BASE_DIR = os.getcwd()
LOCAL_PERSISTENT = os.path.join(BASE_DIR, "local_persistent", "record_archive")
ENV_DATA_DIR = os.getenv("DATA_DIR")

if ENV_DATA_DIR:
    ARCHIVE_DIR = ENV_DATA_DIR
elif os.path.isdir(LOCAL_PERSISTENT):
    ARCHIVE_DIR = LOCAL_PERSISTENT
else:
    ARCHIVE_DIR = os.path.join(BASE_DIR, "record_archive")

print(f"📦 dataset_routes using → {ARCHIVE_DIR}")

WAV_DIR = os.path.join(ARCHIVE_DIR, "wavs")
CSV_PATH = os.path.join(ARCHIVE_DIR, "metadata.csv")
os.makedirs(WAV_DIR, exist_ok=True)

# ✅ Ensure metadata.csv has header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file_name", "text"])


# =====================================================
# ✳️ Helper: append metadata entry
# =====================================================
def append_metadata(file_name: str, text: str):
    """Safely append a new entry to metadata.csv if not already present."""
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
        clean_text = text.strip() if text and text.strip() else "EMPTY_AUDIO"

        if rel_path not in existing:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([rel_path, clean_text])
            print(f"🧾 Added to metadata.csv → {rel_path} | text='{clean_text}'")
        else:
            print(f"⚠️ Duplicate skipped: {rel_path}")

    except Exception as e:
        print(f"❌ Failed to append metadata: {e}")


# =====================================================
# 🟢 /dataset/add — with conversion to PCM WAV
# =====================================================
@router.post("/add")
async def add_sample(file: UploadFile, text: str = Form(...)):
    """
    Add a new manually uploaded sample to the dataset.
    Converts any format (AAC/MP4/WebM/WAV) into true PCM WAV (44.1kHz, mono, 16-bit).
    """
    try:
        # --- Unique filename ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        final_name = f"{base_name}_{timestamp}.wav"
        final_path = os.path.join(WAV_DIR, final_name)

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload.")

        # --- Detect format ---
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

        # --- Decode (pydub first, sf fallback) ---
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

        # --- Convert to PCM RIFF WAV ---
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(
            final_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1"],
        )
        print(f"💾 Saved true PCM WAV → {final_path}")

        # --- Append metadata ---
        append_metadata(final_name, text)

        return {"status": "ok", "file_name": final_name, "text": text.strip()}

    except Exception as e:
        print(f"❌ Failed to add sample: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🟨 Other dataset routes unchanged
# =====================================================
@router.post("/update")
async def update_sample(request: Request):
    ...
@router.post("/delete")
async def delete_sample(request: Request):
    ...
@router.post("/cleanup")
async def cleanup_orphans():
    ...
@router.get("/list")
async def list_samples():
    ...
@router.get("/export")
async def export_dataset():
    ...
