# imports
import os
import uuid
import time
import logging
import requests
import threading
import queue
from pathlib import Path
from datetime import datetime, timezone

# i chose fast api cos its easy to use
# usually i use flask though
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# use pillow for images and exif
from PIL import Image, UnidentifiedImageError, ExifTags

# import the other files
from .database import SessionLocal, init_db
from .models import Image as ImageModel
# sql session
from sqlalchemy.orm import Session
# .env yay
from dotenv import load_dotenv

# provided api url
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
# env token
load_dotenv()
HF_TOKEN = os.getenv("hugapi")

# a new function to caption the image that should either give back a string or nothing
def caption_image(path: Path) -> str | None:
    # if no token, skip
    if not HF_TOKEN:
        logger.warning("HF_API_TOKEN not set — skipping caption")
        return None
    # call api
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        with open(path, "rb") as f:
            resp = requests.post(HF_API_URL, headers=headers, data=f)
        # respond accordingly
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text")
        else:
            logger.error("HF API error %d: %s", resp.status_code, resp.text)
    except Exception as e:
        # handle failures
        logger.exception("HF captioning failed: %s", e)
    return None


# cool little fastapi app
app = FastAPI(title="Image Processing API")

# setup things
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# log uvicorn logs
logger = logging.getLogger("uvicorn.error")

# my storage stuff
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
ORIG_DIR = DATA_DIR / "originals"
THUMB_DIR = DATA_DIR / "thumbnails"
ORIG_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# initialize the db
init_db()

# make the function to get db
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# queue system
task_queue = queue.Queue()

# worker thread
def worker():
    while True:
        image_id = task_queue.get()
        try:
            # process
            process_image_task(image_id)
        except Exception as e:
            # fail pretty
            logger.exception("Worker failed for %s: %s", image_id, e)
        finally:
            # mark task done
            task_queue.task_done()

# start the worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()


# required endpoints!

# post for upload
@app.post("/api/images")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG and PNG supported")

    image_id = str(uuid.uuid4())
    ext = "jpg" if file.content_type == "image/jpeg" else "png"
    saved_path = ORIG_DIR / f"{image_id}.{ext}"

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    img = ImageModel(
        id=image_id,
        original_filename=file.filename,
        original_path=str(saved_path),
        status="queued",
    )
    db.add(img)
    db.commit()

    # enqueue task
    task_queue.put(image_id)

    return {"status": "success", "data": {"image_id": image_id, "status": "queued"}, "error": None}


@app.get("/api/images")
def list_images(db: Session = Depends(get_db)):
    rows = db.query(ImageModel).order_by(ImageModel.created_at.desc()).all()
    out = []
    for r in rows:
        out.append(
            {
                "image_id": r.id,
                "original_name": r.original_filename,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "processed_at": r.processed_at.isoformat() if r.processed_at else None,
            }
        )
    return {"status": "success", "data": out, "error": None}


@app.get("/api/images/{image_id}")
def get_image(image_id: str, db: Session = Depends(get_db)):
    r = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not r:
        raise HTTPException(404, "Image not found")
    return {
        "status": "success",
        "data": {
            "image_id": r.id,
            "original_name": r.original_filename,
            "processed_at": r.processed_at.isoformat() if r.processed_at else None,
            "metadata": {
                "width": r.width,
                "height": r.height,
                "format": r.format,
                "size_bytes": r.size_bytes,
                "exif": r.exif,
            },
            "thumbnails": {
                "small": f"/api/images/{r.id}/thumbnails/small" if r.thumbnail_small_path else None,
                "medium": f"/api/images/{r.id}/thumbnails/medium" if r.thumbnail_medium_path else None,
            },
            "caption": r.caption,
            "status": r.status,
            "error": r.error,
        },
        "error": None,
    }


@app.get("/api/images/{image_id}/thumbnails/{size}")
def get_thumbnail(image_id: str, size: str, db: Session = Depends(get_db)):
    if size not in ("small", "medium"):
        raise HTTPException(400, "size must be 'small' or 'medium'")
    r = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not r:
        raise HTTPException(404, "Image not found")
    path = r.thumbnail_small_path if size == "small" else r.thumbnail_medium_path
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Thumbnail not ready")
    return FileResponse(path)


@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(ImageModel).count()
    done = db.query(ImageModel).filter(ImageModel.status == "done").count()
    failed = db.query(ImageModel).filter(ImageModel.status == "failed").count()
    queued = db.query(ImageModel).filter(ImageModel.status == "queued").count()
    processing = db.query(ImageModel).filter(ImageModel.status == "processing").count()

    avg_ms = None
    rows = db.query(ImageModel).filter(ImageModel.status == "done").all()
    if rows:
        total_ms = sum((r.processing_time_ms or 0) for r in rows)
        avg_ms = total_ms // len(rows)

    return {
        "status": "success",
        "data": {
            "total": total,
            "done": done,
            "failed": failed,
            "queued": queued,
            "processing": processing,
            "avg_processing_time_ms": avg_ms,
        },
        "error": None,
    }

# processing function
def process_image_task(image_id: str):
    db = SessionLocal()
    r = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not r:
        db.close()
        return

    start = time.time()
    try:
        # mark as processing
        r.status = "processing"
        db.add(r)
        db.commit()

        path = Path(r.original_path)
        if not path.exists():
            r.status = "failed"
            r.error = "Original file missing"
            db.add(r)
            db.commit()
            return

        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.size
            fmt = pil_img.format or path.suffix.replace(".", "").upper()
            size_bytes = path.stat().st_size

            # EXIF
            exif_out = None
            try:
                raw_exif = pil_img._getexif()
                if raw_exif:
                    exif_out = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}
            except Exception:
                exif_out = None

            # thumbnails
            small_path = THUMB_DIR / f"{image_id}_small.jpg"
            medium_path = THUMB_DIR / f"{image_id}_medium.jpg"

            thumb_small = pil_img.copy()
            thumb_small.thumbnail((200, 200))
            thumb_small.save(small_path, "JPEG")

            thumb_medium = pil_img.copy()
            thumb_medium.thumbnail((800, 800))
            thumb_medium.save(medium_path, "JPEG")

            # caption
            caption = caption_image(path)

            # update DB
            r.width = width
            r.height = height
            r.format = fmt
            r.size_bytes = size_bytes
            r.exif = exif_out
            r.caption = caption
            r.thumbnail_small_path = str(small_path)
            r.thumbnail_medium_path = str(medium_path)
            r.status = "done"
            r.processed_at = datetime.now(timezone.utc)
            r.processing_time_ms = int((time.time() - start) * 1000)
            db.add(r)
            db.commit()
            logger.info("Processed %s in %d ms", image_id, r.processing_time_ms)

    except UnidentifiedImageError:
        r.status = "failed"
        r.error = "Invalid image file"
        db.add(r)
        db.commit()
    except Exception as e:
        r.status = "failed"
        r.error = str(e)
        db.add(r)
        db.commit()
        logger.exception("Unexpected error processing %s: %s", image_id, e)
    finally:
        db.close()
