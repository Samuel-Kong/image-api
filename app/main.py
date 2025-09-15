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
from fractions import Fraction

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
# redirect for base url
from fastapi.responses import RedirectResponse

# google gemini
import google.generativeai as genai


# env key
load_dotenv()
GEMINI_KEY = os.getenv("geminikey")

# configure gemini if key present
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)


# helpers to clean EXIF values
def parse_exif_value(value):
    if isinstance(value, Fraction):
        try:
            return float(value)
        except Exception:
            return str(value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value.numerator) / float(value.denominator)
        except Exception:
            return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode(errors="ignore")
        except Exception:
            return value.hex()
    if isinstance(value, tuple):
        return [parse_exif_value(v) for v in value]
    return value


def dms_to_deg(dms, ref):
    """Convert GPS degrees/minutes/seconds to decimal degrees."""
    try:
        deg = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
        if ref in ["S", "W"]:
            deg = -deg
        return deg
    except Exception:
        return None


# a new function to caption the image that should either give back a string or nothing
def caption_image(path: Path) -> str | None:
    # if no key, skip
    if not GEMINI_KEY:
        logger.warning("geminikey not set — skipping caption")
        return None
    try:
        # init model
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        # call api with instruction and image
        resp = model.generate_content(
            [
                "You are an image captioning API. Generate a concise caption describing this image clearly. Return only the text that is used to describe the image. NO formatting is needed",
                {"mime_type": "image/jpeg", "data": path.read_bytes()},
            ]
        )
        if resp and resp.text:
            return resp.text.strip()
    except Exception as e:
        # handle failures
        logger.exception("Gemini captioning failed: %s", e)
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

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


# post for upload
@app.post("/api/images")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # only allow jpeg and png
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG and PNG supported")

    # create a new uuid for the image
    image_id = str(uuid.uuid4())
    ext = "jpg" if file.content_type == "image/jpeg" else "png"
    saved_path = ORIG_DIR / f"{image_id}.{ext}"

    # save the file
    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # add to db
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

    # respond success
    return {"status": "success", "data": {"image_id": image_id, "status": "queued"}, "error": None}


# list all images
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


# get details for a single image
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
                "exif": {
                    "camera_make": r.camera_make,
                    "camera_model": r.camera_model,
                    "datetime_original": r.datetime_original,
                    "lens_model": r.lens_model,
                    "iso": r.iso,
                    "focal_length": r.focal_length,
                    "exposure_time": r.exposure_time,
                    "gps": {"lat": r.gps_lat, "lon": r.gps_lon},
                    "raw": r.exif_json,
                },
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


# get thumbnail by size
@app.get("/api/images/{image_id}/thumbnails/{size}")
def get_thumbnail(image_id: str, size: str, db: Session = Depends(get_db)):
    # only allow small or medium
    if size not in ("small", "medium"):
        raise HTTPException(400, "size must be 'small' or 'medium'")
    r = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not r:
        raise HTTPException(404, "Image not found")
    path = r.thumbnail_small_path if size == "small" else r.thumbnail_medium_path
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Thumbnail not ready")
    return FileResponse(path)


# get stats for all images
@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(ImageModel).count()
    done = db.query(ImageModel).filter(ImageModel.status == "done").count()
    failed = db.query(ImageModel).filter(ImageModel.status == "failed").count()
    queued = db.query(ImageModel).filter(ImageModel.status == "queued").count()
    processing = db.query(ImageModel).filter(ImageModel.status == "processing").count()

    # calculate average processing time
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

        # check if file exists
        path = Path(r.original_path)
        if not path.exists():
            r.status = "failed"
            r.error = "Original file missing"
            db.add(r)
            db.commit()
            return

        # open and process image
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.size
            fmt = pil_img.format or path.suffix.replace(".", "").upper()
            size_bytes = path.stat().st_size

            # EXIF
            exif_out = {}
            camera_make = camera_model = datetime_original = lens_model = None
            iso = focal_length = exposure_time = None
            gps_lat = gps_lon = None

            try:
                exif_data = pil_img.getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        value = parse_exif_value(value)
                        exif_out[str(tag)] = value

                        if tag == "Make":
                            camera_make = value
                        elif tag == "Model":
                            camera_model = value
                        elif tag == "DateTimeOriginal":
                            datetime_original = value
                        elif tag == "LensModel":
                            lens_model = value
                        elif tag == "ISOSpeedRatings":
                            try:
                                iso = int(value)
                            except Exception:
                                iso = None
                        elif tag == "FocalLength":
                            try:
                                focal_length = float(value)
                            except Exception:
                                focal_length = None
                        elif tag == "ExposureTime":
                            exposure_time = str(value)
                        elif tag == "GPSInfo":
                            gps_data = value
                            if isinstance(gps_data, dict):
                                lat = lon = None
                                if 2 in gps_data and 1 in gps_data:
                                    lat = dms_to_deg(gps_data[2], gps_data[1])
                                if 4 in gps_data and 3 in gps_data:
                                    lon = dms_to_deg(gps_data[4], gps_data[3])
                                gps_lat, gps_lon = lat, lon
            except Exception as e:
                logger.warning("Could not extract EXIF for %s: %s", image_id, e)
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

            r.camera_make = camera_make
            r.camera_model = camera_model
            r.datetime_original = datetime_original
            r.lens_model = lens_model
            r.iso = iso
            r.focal_length = focal_length
            r.exposure_time = exposure_time
            r.gps_lat = gps_lat
            r.gps_lon = gps_lon
            r.exif_json = exif_out

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
        # handle invalid image
        r.status = "failed"
        r.error = "Invalid image file"
        db.add(r)
        db.commit()
    except Exception as e:
        # handle unexpected errors
        r.status = "failed"
        r.error = str(e)
        db.add(r)
        db.commit()
        logger.exception("Unexpected error processing %s: %s", image_id, e)
    finally:
        db.close()
