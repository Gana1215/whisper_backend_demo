# ===============================================
# 📚 dataset_routes.py (v3.2 — PCM Stable + Re-Record + Export)
# ✅ Converts all uploads to RIFF PCM WAV (mono, 44.1kHz, 16-bit)
# ✅ Fixes /list and naming issues
# ✅ Adds /update_audio (Re-Record)
# ✅ Adds /export (ZIP Download)
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
# 🟢 /dataset/add — Converts & saves as PCM WAV
# =====================================================
@router.post("/add")
async def add_sample(file: UploadFile, text: str = Form(...)):
    try:
        # --- Sanitize & fix filename ---
        base = os.path.splitext(os.path.basename(file.filename))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base.startswith("usr001_") and len(base.split("_")) >= 3:
            final_name = f"{base}.wav"  # already has timestamp
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
# 🔁 /dataset/update_audio — Re-record existing row
# =====================================================
@router.post("/update_audio")
async def update_audio(file: UploadFile, file_name: str = Form(...)):
    """
    Replace an existing WAV entry with a new recording.
    Keeps the same file_name and metadata text intact.
    Converts the new audio to true PCM RIFF WAV (44.1kHz mono 16-bit).
    """
    try:
        target_path = os.path.join(WAV_DIR, os.path.basename(file_name))
        if not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail=f"{file_name} not found")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload.")

        # --- Decode and normalize ---
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

        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio.export(
            target_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1"],
        )

        print(f"🔁 Re-record replaced → {target_path}")
        return {"status": "ok", "message": f"{file_name} updated successfully"}

    except Exception as e:
        print(f"❌ Failed to update audio: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 🧾 /dataset/list — Fixed version
# =====================================================
@router.get("/list")
async def list_samples():
    try:
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

        return {"count": len(rows), "samples": rows}

    except Exception as e:
        print(f"❌ /dataset/list failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================
# 📦 /dataset/export — Downloadable ZIP
# =====================================================
@router.get("/export")
async def export_dataset():
    """
    Create and return a downloadable ZIP of the dataset (WAVs + metadata.csv).
    Works globally (browser-initiated download supported).
    """
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
            media_type="application/zip",
        )
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
