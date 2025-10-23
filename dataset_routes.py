# ===============================================
# 📚 dataset_routes.py (patched 2025-10-22)
# ✅ JSON-compatible update/delete for new frontend
# ✅ Simple CSV-based dataset manager for Mongolian Whisper
# ✅ Handles add / update / delete / list operations
# ✅ All samples stored under: record_archive/wavs/
# ✅ Metadata file: record_archive/metadata.csv
# ===============================================

import os
import csv
import aiofiles
from fastapi import APIRouter, UploadFile, Form, Body, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/dataset", tags=["dataset"])

# --- Paths ---
DATA_DIR = "record_archive"
WAV_DIR = os.path.join(DATA_DIR, "wavs")
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")

# --- Ensure directories exist ---
os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Ensure CSV header exists ---
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])

# 🟩 Add new sample (upload WAV + text)
@router.post("/add")
async def add_sample(file: UploadFile, text: str = Form(...)):
    try:
        file_path = os.path.join(WAV_DIR, file.filename)
        # Save WAV asynchronously
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file.read())

        # Append entry to CSV
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([file.filename, text])

        return {"status": "ok", "message": "Added", "file_name": file.filename}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟨 Update text for an existing sample (now JSON body)
@router.post("/update")
async def update_sample(data: dict = Body(...)):
    file_name = data.get("file_name")
    new_text = data.get("new_text")
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name missing")

    try:
        rows, found = [], False
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["file_name"] == file_name:
                    r["text"] = new_text or ""
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

# 🟥 Delete sample (now JSON body)
@router.post("/delete")
async def delete_sample(data: dict = Body(...)):
    file_name = data.get("file_name")
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name missing")

    try:
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

        wav_path = os.path.join(WAV_DIR, file_name)
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return {"status": "ok", "message": f"Deleted {file_name}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟦 List all dataset samples
@router.get("/list")
async def list_samples():
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return {"count": len(data), "samples": data}
    except FileNotFoundError:
        return {"count": 0, "samples": []}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
