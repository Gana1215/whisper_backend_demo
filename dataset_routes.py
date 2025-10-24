# ===============================================
# 📚 dataset_routes.py (v1.8 — Orphan Cleanup Edition)
# ✅ Compatible with both JSON & FormData
# ✅ CSV header protection (auto-restores if missing)
# ✅ Prevents duplicate entries
# ✅ Removes orphaned temporary files safely
# ✅ Keeps only usr001_*.wav dataset files
# ===============================================

import os
import csv
import aiofiles
from fastapi import APIRouter, UploadFile, Form, Request, HTTPException
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
        csv.writer(f).writerow(["file_name", "text"])

# 🟩 Add new sample
@router.post("/add")
async def add_sample(file: UploadFile, text: str = Form(...)):
    try:
        file_path = os.path.join(WAV_DIR, file.filename)

        if os.path.exists(file_path):
            return JSONResponse(status_code=400, content={"error": f"{file.filename} exists."})

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file.read())

        # Ensure header
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["file_name", "text"])

        # Append record
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([file.filename.strip(), text.strip()])

        return {"status": "ok", "message": "Added", "file_name": file.filename}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟨 Update sample text
@router.post("/update")
async def update_sample(request: Request):
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

# 🟥 Delete one sample
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

# 🧹 Clean orphan files (usr_*, wv_*, temp_*)
@router.post("/cleanup")
async def cleanup_orphans():
    try:
        if not os.path.exists(WAV_DIR):
            return {"status": "ok", "removed": 0, "message": "No wavs folder"}

        removed = 0
        for f in os.listdir(WAV_DIR):
            # only remove files that are not in CSV AND start with 'usr_' or 'wv_'
            if not f.endswith(".wav"):
                continue
            if f.startswith(("usr_", "wv_")) and not f.startswith("usr001_"):
                file_path = os.path.join(WAV_DIR, f)
                os.remove(file_path)
                removed += 1

        return {"status": "ok", "removed": removed, "message": f"🧹 Removed {removed} orphan files"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🟦 List all dataset samples
@router.get("/list")
async def list_samples():
    try:
        if not os.path.exists(CSV_PATH):
            return {"count": 0, "samples": []}

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = []
            for r in reader:
                clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
                data.append(clean)

        return {"count": len(data), "samples": data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
