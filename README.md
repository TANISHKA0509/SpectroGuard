# SpectroGuard — Deepfake Video Call Detection System

> **Educational prototype.** Analyzes uploaded videos and classifies them as
> **REAL** or **FAKE** using a publicly available pre-trained deepfake
> detection model (FaceForge Xception, trained on FaceForensics++).

SpectroGuard is a full-stack prototype built around **software engineering
fundamentals**: computer vision (OpenCV), a pre-trained deep-learning model
(PyTorch), a clean REST API (FastAPI), asynchronous job processing, and a
responsive frontend (HTML/CSS/JavaScript). It is **not** a research-grade or
production-level detector — it is designed to demonstrate how the pieces fit
together and to serve as a portfolio project.

![stack](https://img.shields.io/badge/stack-FastAPI-009688) ![ml](https://img.shields.io/badge/ml-PyTorch%20%7C%20OpenCV-EE4C2C) ![license](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **Upload MP4** (also `.mov`, `.avi`, `.mkv`, `.webm`; max 200 MB) via a
  drag-and-drop web UI or the REST API.
- **Frame sampling** — instead of classifying every frame, the pipeline
  samples every *n*-th frame (configurable, default 1-in-10, max 30 frames)
  so analysis stays fast on CPU.
- **Face-aware pre-processing** — OpenCV Haar cascade crops the dominant face
  (falls back to the full frame when no face is detected).
- **Pre-trained model, inference only** — the model is downloaded once from
  the Hugging Face Hub and loaded at application startup. No training code.
- **Aggregated verdict** — per-frame predictions are combined with
  **majority voting**; the returned confidence is the average probability of
  the winning class across sampled frames.
- **Background processing** — analysis runs in a worker thread; the UI shows
  live status/progress via polling.
- **Clean result** — prediction badge, confidence bar, frames analyzed, votes,
  faces found and processing time.
- **Error handling** — unsupported formats, corrupt videos and oversized
  uploads return clear messages.

---

## 🧠 Model

| Field        | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Checkpoint   | [`huzaifanasirrr/faceforge-detector`](https://huggingface.co/huzaifanasirrr/faceforge-detector) |
| Backbone     | XceptionNet (timm), 20.8 M params                                     |
| Head         | Dropout → Linear(2048→512) → ReLU → Dropout → Linear(512→2)          |
| Dataset      | FaceForensics++ (c40) — Real vs. Fake faces                           |
| Input        | 224×224 RGB, normalized with mean = std = 0.5 (→ [-1, 1])             |
| Classes      | index 0 → `REAL`, index 1 → `FAKE`                                    |
| Size         | ~250 MB, downloaded automatically into `models/` on first run          |

The weights file is ignored by git (see `.gitignore`) and fetched at startup
or at Docker build time, so the repository stays small.

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │                    Browser                        │
                    │   index.html · style.css · script.js (frontend)  │
                    └───────────────┬──────────────────────────────────┘
                                    │  REST (JSON) / static files
                    ┌───────────────▼──────────────────────────────────┐
                    │                  FastAPI (app.py)                │
                    │  upload → validate → save → job store (in-mem)   │
                    │  /api/analyze spawns a background worker thread  │
                    └───────────────┬──────────────────────────────────┘
                                    │
            ┌───────────────────────▼───────────────────────────────┐
            │                 Prediction pipeline                    │
            │   video_processor.py   OpenCV frame extraction +      │
            │                        every-nth sampling + face crop │
            │   inference.py         preprocess → model → softmax   │
            │                        → majority vote aggregation    │
            │   model.py             pre-trained Xception (timm)    │
            └───────────────────────────────────────────────────────┘
```

**Data flow per video**

1. `POST /api/upload` stores the file under `uploads/` and registers a job.
2. `POST /api/analyze/{id}` launches a background worker.
3. `video_processor.extract_frames` samples frames (`1 in 10`, max 30).
4. Each frame is optionally face-cropped, resized to 224×224, normalized.
5. `inference.run_inference` runs the model over the batched frames.
6. `aggregate_video_prediction` applies majority voting + average confidence.
7. The UI polls `GET /api/status/{id}` and renders `GET /api/result/{id}`.

---

## 📁 Folder structure

```
spectroguard/
├── backend/
│   ├── app.py              # FastAPI app, routes, job store, static serving
│   ├── model.py            # pre-trained model loading + config
│   ├── inference.py        # batched inference + aggregation
│   ├── video_processor.py  # OpenCV frame extraction / sampling / cropping
│   └── utils.py            # shared helpers (ids, extensions, timing)
├── frontend/
│   ├── index.html          # page structure
│   ├── style.css           # styling
│   └── script.js           # upload / analyze / poll / render logic
├── models/                 # downloaded weights live here (git-ignored)
├── uploads/                # uploaded videos (git-ignored)
├── scripts/
│   ├── download_model.py   # fetch weights ahead of time
│   └── create_test_video.py# generate a synthetic MP4 for smoke tests
├── requirements.txt
├── Dockerfile              # container image (Render / HF Spaces / local)
├── render.yaml             # Render.com free-tier deploy config
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python **3.10 – 3.12**
- (Optional) GPU — the code auto-detects CUDA, but CPU is fully supported.

### 1. Clone & create a virtual environment

```bash
git clone <your-repo-url> spectroguard
cd spectroguard
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# CPU-only PyTorch (recommended - much smaller download)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> If you have a CUDA GPU and want GPU inference, install the CUDA builds of
> torch/torchvision from https://pytorch.org instead.

### 3. Download the pre-trained model

```bash
python scripts/download_model.py     # ~250 MB into models/
```

This also runs automatically on the first server start, so this step is
optional locally.

---

## 🚀 How to run

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — upload a video, click **Analyze Video**, and
watch the status progress to a REAL/FAKE verdict.

**Smoke-test the pipeline with a generated video:**

```bash
python scripts/create_test_video.py   # writes uploads/test_sample.mp4
```

> The generated clip contains moving shapes, **not a real face**, so its
> verdict is meaningless for deepfake evaluation — it only verifies that the
> whole pipeline works.

### API documentation

FastAPI auto-generates interactive docs at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🔌 API endpoints

| Method | Path                            | Description                                     |
|--------|---------------------------------|-------------------------------------------------|
| GET    | `/`                             | Frontend UI                                     |
| GET    | `/health`                       | Health / readiness check (used by platforms)    |
| POST   | `/api/upload`                   | Upload a video file (multipart `file`)          |
| POST   | `/api/analyze/{video_id}`       | Start background analysis                       |
| GET    | `/api/status/{video_id}`        | Job status + progress (0–100%)                  |
| GET    | `/api/result/{video_id}`        | Final prediction (when `status == done`)        |
| GET    | `/api/videos`                   | List all jobs in this process                   |
| GET    | `/api/videos/{video_id}/video`  | Stream the uploaded video for preview           |
| DELETE | `/api/videos/{video_id}`        | Delete an upload + job                          |

### Example result JSON

```json
{
  "video_id": "6d72b10a325a",
  "result": {
    "prediction": "FAKE",
    "confidence": 0.8749,
    "frames_analyzed": 4,
    "votes": { "REAL": 0, "FAKE": 4 },
    "average_confidence_real": 0.1251,
    "average_confidence_fake": 0.8749,
    "processing_time": 3.84,
    "inference_time": 3.83,
    "total_frames": 36,
    "fps": 12.0,
    "frames_sampled": 4,
    "sampling_ratio": "1 in 10",
    "faces_detected": 0,
    "model": "huzaifanasirrr/faceforge-detector"
  }
}
```

**Quick curl example**

```bash
# upload
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/upload

# analyze (use the returned video_id)
curl -X POST http://localhost:8000/api/analyze/<video_id>

# poll until done, then fetch the result
curl http://localhost:8000/api/result/<video_id>
```

---

## 🐳 Running with Docker

```bash
docker build -t spectroguard .
docker run -p 7860:7860 spectroguard
# -> http://localhost:7860
```

The image installs CPU-only PyTorch and downloads the model at build time, so
it starts fast.

---

## ☁️ Deployment

The app is a standard FastAPI service and deploys anywhere that runs Python.
Two ready-made options:

### Option A — Render (free tier)

1. Push this repo to GitHub.
2. On [Render](https://render.com) → **New → Web Service** → connect the repo.
3. Render detects `render.yaml` automatically (Docker runtime).
4. Replace `YOUR_GITHUB_USERNAME` in `render.yaml` (or just let Render generate
   the service manually with **Docker** runtime and health check path `/health`).
5. Deploy → you get a public URL like `https://spectroguard.onrender.com`.

### Option B — Hugging Face Spaces (great for ML demos)

1. Create a Space on https://huggingface.co/new-space with **SDK: Docker**.
2. Push this repo (including the `Dockerfile`).
3. HF Spaces builds the image (model is fetched at build) and serves it on
   port `7860`, giving you a public URL `https://<user>-spectroguard.hf.space`.

> **Note:** CPU inference takes a few seconds per video (a 30-frame sample ≈
> 3–6 s). Free tiers are fine for demos.

---

## 🛠️ Technologies

| Layer      | Technology                                    |
|------------|-----------------------------------------------|
| Backend    | Python, FastAPI, Uvicorn, python-multipart    |
| ML         | PyTorch, TorchVision, timm, Hugging Face Hub  |
| CV         | OpenCV (frame extraction, Haar face detection)|
| Maths      | NumPy                                         |
| Frontend   | HTML, CSS, vanilla JavaScript (no build step) |
| DevOps     | Docker, Render, Hugging Face Spaces           |

---

## ⚠️ Limitations

- **Educational scope** — this is a prototype, not a production/forensic tool.
- **Single-frame analysis** — no temporal/optical-flow features; frames are
  judged independently and merged by voting.
- **Domain** — the model was trained on FaceForensics++ (c40) faces; accuracy
  drops on other manipulation methods, heavy compression, unusual angles or
  videos without clear faces.
- **Face detection** — relies on OpenCV's Haar cascade (fast but imprecise;
  falls back to full-frame crops).
- **In-memory job store** — jobs and uploads vanish on restart; single-process
  only, no queue/worker scaling.
- **CPU latency** — a full run takes seconds on CPU; free cloud tiers share
  CPU with other tenants.
- **Security** — uploads are stored unencrypted on the server; no auth.

---

## 🔮 Future improvements

- [ ] Face alignment + cropping via a more robust detector (MTCNN / YuNet).
- [ ] Temporal aggregation (LSTM / voting over sliding windows of frames).
- [ ] Multi-model ensemble (e.g., MesoNet + EfficientNet) with per-model scores.
- [ ] Stream/live-video support using WebSockets for real "video-call" checks.
- [ ] Persistent storage + a lightweight DB for job history.
- [ ] Task queue (Celery/RQ) + worker pool for horizontal scaling.
- [ ] Rate limiting, auth, and secure file handling for production use.
- [ ] Exportable per-frame heat-maps showing *where* the model looks.

---

## 📚 References

- XceptionNet — Chollet (2017): *"Xception: Deep Learning with Depthwise Separable Convolutions"*
- FaceForensics++ — Rössler et al. (2019)
- Pre-trained model: [FaceForge Detector](https://huggingface.co/huzaifanasirrr/faceforge-detector)

## 📄 License

MIT — see the LICENSE note in the model card. Use responsibly: this tool is
for education and legitimate verification, not surveillance or abuse.
