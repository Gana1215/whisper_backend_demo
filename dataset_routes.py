# ===============================================
# 📚 dataset_routes.py (v2.4 — Persistent Unified Edition)
# ✅ Always shares dataset folder with app.py
# ✅ Prefers local_persistent/record_archive if it exists
# ✅ Honors DATA_DIR (Render/local override)
# ✅ Auto-creates metadata.csv with header
# ✅ Prevents duplicates, cleans orphans
# ✅ Compatible with FormData & JSON
# ===============================================

import os
import csv
import aiofiles
from fastapi import APIRouter, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/dataset", tags=["dataset"])

# =====================================================
# 🌐 Resolve shared persistent dataset directory
# =====================================================
# app.py does: IS_RENDER = os.path.exists("/opt/render"), BASE_DIR=...; os.chdir(BASE_DIR)
# At this import time, cwd should already be BASE_DIR.
BASE_DIR = os.getcwd()

LOCAL_PERSISTENT = os.path.join(BASE_DIR, "local_persistent", "record_archive")
ENV_DATA_DIR = os.getenv("DATA_DIR")

if ENV_DATA_DIR:
    ARCHIVE_DIR = ENV_DATA_DIR
elif os.path.isdir(LOCAL_PERSISTENT):
    # Prefer local_persistent if present (local dev)
    ARCHIVE_DIR = LOCAL_PERSISTENT
else:
    # Fallback to project-relative record_archive
    ARCHIVE_DIR = os.path.join(BASE_DIR, "record_archive")

print(f"📦 dataset_routes using → {ARCHIVE_DIR}")

WAV_DIR = os.path.join(ARCHIVE_DIR, "wavs")
CSV_PATH = os.path.join(ARCHIVE_DIR, "metadata.csv")
os.makedirs(WAV_DIR, exist_ok=True)

# ✅ Ensure metadata.csv has header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file_name", "text"])

# 🟩 Add new sample -------------------------------------------------
@router.post("/add")
async def add_sample(file: UploadFile, text: str = Form(...)):
    """Add a new manually uploaded sample to the dataset"""
    try:
        file_path = os.path.join(WAV_DIR, file.filename)

        # prevent overwriting
        if os.path.exists(file_path):
            return JSONResponse(status_code=400, content={"error": f"{file.filename} already exists."})

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file.read())

        # ensure header
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["file_name", "text"])

        # append record (relative path for portability)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"wavs/{file.filename.strip()}", text.strip()])

        return {"status": "ok", "message": "Added", "file_name": file.filename}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟨 Update existing sample -----------------------------------------
@router.post("/update")
async def update_sample(request: Request):
    """Update the text for a given WAV file"""
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            file_name = data.get("file_name")
            new_text = data.get("new_text", "")
        else:
            form = await request.form()
            file_name = form.get("file_name")
            new_text = form.get("new_text", "")

        if not file_name:
            raise HTTPException(status_code=400, detail="file_name missing")

        rows, found = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["file_name"] == file_name:
                    r["text"] = new_text.strip()
                    found = True
                rows.append(r)

        if not found:
            return JSONResponse(status_code=404, content={"error": f"{file_name} not found"})

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)

        return {"status": "ok", "message": f"Updated {file_name}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟥 Delete sample --------------------------------------------------
@router.post("/delete")
async def delete_sample(request: Request):
    """Delete a dataset entry and its corresponding WAV file"""
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            file_name = data.get("file_name")
        else:
            form = await request.form()
            file_name = form.get("file_name")

        if not file_name:
            raise HTTPException(status_code=400, detail="file_name missing")

        rows, found = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["file_name"] == file_name:
                    found = True
                    continue
                rows.append(r)

        if not found:
            return JSONResponse(status_code=404, content={"error": f"{file_name} not found"})

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)

        wav_path = os.path.join(WAV_DIR, os.path.basename(file_name))
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return {"status": "ok", "message": f"Deleted {file_name}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🧹 Clean orphaned files -------------------------------------------
@router.post("/cleanup")
async def cleanup_orphans():
    """Remove orphaned temporary or unlisted recordings"""
    try:
        if not os.path.exists(WAV_DIR):
            return {"status": "ok", "removed": 0, "message": "No wavs folder"}

        removed = 0
        valid_files = set()

        # load filenames from CSV
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    valid_files.add(os.path.basename(row["file_name"]))

        for f in os.listdir(WAV_DIR):
            if not f.endswith(".wav"):
                continue
            # remove unlisted or temporary recordings
            if (f.startswith(("usr_", "wv_", "temp_")) and not f.startswith("usr001_")) \
               or f not in valid_files:
                os.remove(os.path.join(WAV_DIR, f))
                removed += 1

        return {"status": "ok", "removed": removed, "message": f"🧹 Removed {removed} orphan files"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟦 List samples ----------------------------------------------------
@router.get("/list")
async def list_samples():
    """List all samples from metadata.csv"""
    try:
        if not os.path.exists(CSV_PATH):
            return {"count": 0, "samples": []}

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = [
                {k.strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in r.items() if k}
                for r in reader
            ]

        return {"count": len(data), "samples": data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
