"""SpectroGuard -- FastAPI backend.

REST API that accepts a video upload, runs the pre-trained deepfake detector
over sampled frames in the background, and returns a Real/Fake verdict with a
confidence score.

The frontend (``frontend/``) is served from the same app, so the whole
prototype runs on a single server.
"""

from __future__ import annotations

import logging
import mimetypes
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import model as model_module
from .inference import aggregate_video_prediction, run_inference
from .utils import (
    ALLOWED_EXTENSIONS,
    ensure_dir,
    generate_video_id,
    is_allowed_file,
    safe_extension,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths & limits
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ensure_dir(BASE_DIR / "uploads")
FRONTEND_DIR = BASE_DIR / "frontend"

MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

#: In-memory job store: video_id -> job dict.
#: (In-memory is fine for a prototype; see README limitations section.)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# App lifecycle: load the pre-trained model once at startup.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    model, device = model_module.load_model()
    app.state.model = model
    app.state.device = device
    logger.info("SpectroGuard ready (device=%s)", device)
    yield


app = FastAPI(
    title="SpectroGuard",
    description="Deepfake Video Call Detection -- educational prototype using a "
    "pre-trained FaceForensics++ Xception model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def get_job(video_id: str) -> dict:
    """Return the job entry or raise 404."""
    with JOBS_LOCK:
        job = JOBS.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return job


def _process_video(video_id: str) -> None:
    """Background worker: extract -> preprocess -> infer -> aggregate."""
    job = get_job(video_id)
    start = time.perf_counter()
    try:
        job["message"] = "Sampling frames and running the model..."
        output = run_inference(
            app.state.model,
            app.state.device,
            job["filepath"],
        )
        job["progress"] = 80
        job["message"] = "Aggregating frame predictions..."

        result = aggregate_video_prediction(
            probabilities=output["probabilities"],
            frame_predictions=output["frame_predictions"],
            frame_confidences=output["frame_confidences"],
        )
        # Extend with video-level metadata
        result["processing_time"] = round(time.perf_counter() - start, 2)
        result["inference_time"] = output["inference_time"]
        result.update(output["metadata"])
        result["faces_detected"] = output["face_found"]
        result["model"] = model_module.MODEL_REPO_ID

        job["result"] = result
        job["progress"] = 100
        job["status"] = "done"
        job["message"] = "Analysis complete."
    except Exception as exc:  # noqa: BLE001 -- surface any processing failure
        logger.exception("Analysis failed for video %s", video_id)
        job["status"] = "error"
        job["message"] = str(exc)
        job["error"] = str(exc)


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness + readiness check used by deployment platforms."""
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "device": str(getattr(app.state, "device", "n/a")),
        "model": model_module.MODEL_REPO_ID,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Accept a video file, store it under uploads/, and register a job."""
    filename = file.filename or "video.mp4"
    if not is_allowed_file(filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{Path(filename).suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    video_id = generate_video_id()
    dest = UPLOADS_DIR / f"{video_id}{safe_extension(filename)}"

    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (max {MAX_UPLOAD_MB} MB)",
                )
            out.write(chunk)

    with JOBS_LOCK:
        JOBS[video_id] = {
            "video_id": video_id,
            "original_filename": filename,
            "stored_filename": dest.name,
            "filepath": str(dest),
            "size_bytes": size,
            "status": "uploaded",
            "message": "Uploaded. Ready to analyze.",
            "progress": 0,
            "created_at": time.time(),
            "result": None,
            "error": None,
        }

    logger.info("Uploaded '%s' as %s (%d bytes)", filename, dest.name, size)
    return {
        "video_id": video_id,
        "filename": filename,
        "size_bytes": size,
        "status": "uploaded",
        "video_url": f"/api/videos/{video_id}/video",
    }


@app.post("/api/analyze/{video_id}")
def start_analysis(video_id: str) -> dict:
    """Kick off analysis in a background thread."""
    job = get_job(video_id)
    if job["status"] in ("processing", "done"):
        raise HTTPException(status_code=409, detail=f"Analysis already {job['status']}")
    job["status"] = "processing"
    job["message"] = "Starting analysis..."
    job["progress"] = 5
    threading.Thread(target=_process_video, args=(video_id,), daemon=True).start()
    return {"video_id": video_id, "status": "processing"}


@app.get("/api/status/{video_id}")
def get_status(video_id: str) -> dict:
    """Poll the job status / progress."""
    job = get_job(video_id)
    return {
        "video_id": video_id,
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
        "error": job["error"],
        "elapsed": round(time.time() - job["created_at"], 1),
    }


@app.get("/api/result/{video_id}")
def get_result(video_id: str) -> dict:
    """Return the final prediction when analysis is done."""
    job = get_job(video_id)
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Analysis not finished yet")
    return {"video_id": video_id, "result": job["result"]}


@app.get("/api/videos")
def list_videos() -> dict:
    """List all jobs known to this process."""
    with JOBS_LOCK:
        videos = [
            {
                "video_id": j["video_id"],
                "filename": j["original_filename"],
                "status": j["status"],
                "prediction": (j["result"] or {}).get("prediction"),
                "created_at": j["created_at"],
            }
            for j in JOBS.values()
        ]
    return {"videos": videos}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    """Delete an uploaded video (and its job entry)."""
    job = get_job(video_id)
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete while processing")
    Path(job["filepath"]).unlink(missing_ok=True)
    with JOBS_LOCK:
        JOBS.pop(video_id, None)
    return {"deleted": video_id}


@app.get("/api/videos/{video_id}/video")
def get_video_file(video_id: str) -> FileResponse:
    """Stream the uploaded video back to the browser for preview."""
    job = get_job(video_id)
    media_type = mimetypes.guess_type(job["stored_filename"])[0] or "video/mp4"
    return FileResponse(job["filepath"], media_type=media_type)


# --------------------------------------------------------------------------
# Serve the static frontend (mounted last so /api/* routes take precedence)
# --------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
