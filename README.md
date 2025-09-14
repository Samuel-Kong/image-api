# 🖼️ Image Processing API
### Written by GPT-4.1
A FastAPI-based service for uploading, processing, and analyzing images.  
Features include thumbnail generation, structured metadata extraction, and image captioning via **Google Gemini 2.0 Flash Lite**.  
Built with **FastAPI + SQLite + PIL** and a simple **queue worker system** (FIFO, one task at a time).

---

## 🚀 Features

- Upload **JPG/PNG** images via REST API
- Store images & metadata in **SQLite**
- Extract:
  - Width, height, format, file size
  - **Structured EXIF metadata** (camera, ISO, GPS, etc.)
  - Raw EXIF JSON dump
- Generate **small** (200px) and **medium** (800px) thumbnails
- AI captions via [Google Generative AI](https://ai.google.dev/)
- Built-in **queue system** so tasks run sequentially, one at a time
- REST endpoints with OpenAPI docs (`/docs`)

---

## 📂 Project Structure

```

image-api/
│── app/
│   ├── main.py        # FastAPI app, endpoints, worker queue
│   ├── models.py      # SQLAlchemy ORM models
│   ├── database.py    # SQLite session & init
│── requirements.txt   # Dependencies
│── README.md          # Documentation

````

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/Samuel-Kong/image-api.git
cd image-api
````

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

* **Windows PowerShell**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Command Prompt**

  ```cmd
  .venv\Scripts\activate.bat
  ```
* **Linux/Mac**

  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a `.env` file in the project root:

```
geminikey=your_google_generative_ai_key_here
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

The app automatically loads this via **python-dotenv**.

---

## ▶️ Run the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server runs at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)
Interactive API docs:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API Endpoints

### Upload Image

`POST /api/images`
Upload a JPG/PNG file. Returns `image_id` and `status=queued`.

---

### Get All Images

`GET /api/images`
List all uploaded images with statuses.

---

### Get Image Metadata

`GET /api/images/{image_id}`
Retrieve metadata, thumbnails, AI caption, and status.

Structured `exif` block includes:

* `camera_make`
* `camera_model`
* `datetime_original`
* `lens_model`
* `iso`
* `focal_length`
* `exposure_time`
* `gps { lat, lon }`
* `raw` (full EXIF JSON dump)

---

### Get Thumbnail

`GET /api/images/{image_id}/thumbnails/{size}`

* `size=small` → 200px
* `size=medium` → 800px

---

### Stats

`GET /api/stats`
Returns totals, status counts, and average processing time.

---

## 🔄 Queue System

Uploads don’t process instantly. Instead:

1. `POST /api/images` → file is saved, DB record created with `status="queued"`.
2. A background **worker thread** picks tasks one by one (FIFO).
3. Status flow:
   `queued → processing → done/failed`
4. Poll `GET /api/images/{image_id}` to check progress.

This ensures **only one image is processed at a time**, preventing overload.

---

## 🛠️ Tech Stack

* **FastAPI** — REST framework
* **SQLite + SQLAlchemy** — lightweight database
* **Pillow (PIL)** — image processing
* **Google Generative AI (Gemini 2.0 Flash Lite)** — AI captions
* **Queue + Worker Thread** — sequential task execution
* **python-dotenv** — environment variable management

---

## 📝 Example Workflow

1. Upload an image:

   ```bash
   curl -X POST "http://127.0.0.1:8000/api/images" -F "file=@cat.jpg"
   ```

   Response:

   ```json
   { "status": "success", "data": { "image_id": "1234", "status": "queued" }, "error": null }
   ```

2. Check status:

   ```bash
   curl http://127.0.0.1:8000/api/images/1234
   ```

3. Get thumbnail:

   ```bash
   curl http://127.0.0.1:8000/api/images/1234/thumbnails/small --output thumb.jpg
   ```

---

## 📜 License

MIT License © 2025 Samuel Kong
