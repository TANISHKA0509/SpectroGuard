"""SpectroGuard -- FastAPI backend.

REST API that accepts a video upload, runs the pre-trained deepfake detector
over sampled frames in the background, and returns a Real/Fake verdict with a
confidence score.

Also includes simple email/password authentication (SQLite + JWT):
the first ``FREE_UPLOADS`` checks are free for anonymous clients; afterwards
uploading requires a signed-in account.

The frontend (``frontend/``) is served from the same app, so the whole
prototype runs on a single server.
"""

from __future__ import annotations

import logging
import mimetypes
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth as auth_module
from . import database as db
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

#: Number of free anonymous uploads before sign-in is required.
FREE_UPLOADS = 3

#: In-memory job store: video_id -> job dict.
#: (In-memory is fine for a prototype; see README limitations section.)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthRequest(BaseModel):
    """Body for register/login requests."""

    email: str
    password: str


# --------------------------------------------------------------------------
# App lifecycle: init DB and load the pre-trained model once at startup.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    model, device = model_module.load_model()
    app.state.model = model
    app.state.device = device
    logger.info("SpectroGuard ready (device=%s)", device)
    yield


app = FastAPI(
    title="SpectroGuard",
    description="Deepfake Video Call Detection -- educational prototype using a "
    "pre-trained FaceForensics++ Xception model.",
    version="1.1.0",
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


def current_user_or_none(authorization: str | None) -> dict | None:
    """Resolve the Authorization header to a user dict (or None)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    user_id = auth_module.decode_token(token)
    if user_id is None:
        return None
    return db.get_user_by_id(user_id)


def client_id_or_default(x_client_id: str | None) -> str:
    return (x_client_id or "anonymous").strip() or "anonymous"


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
# Auth API
# --------------------------------------------------------------------------
@app.post("/api/auth/register")
def register(payload: AuthRequest) -> dict:
    """Create an account and return a token."""
    email = payload.email.strip().lower()
    if not EMAIL_RE.fullmatch(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user_id = uuid.uuid4().hex
    salt, password_hash = auth_module.hash_password(payload.password)
    db.create_user(user_id, email, password_hash, salt)

    token = auth_module.create_token(user_id)
    logger.info("Registered new user %s", email)
    return {"token": token, "user": {"id": user_id, "email": email}}


@app.post("/api/auth/login")
def login(payload: AuthRequest) -> dict:
    """Verify credentials and return a token."""
    email = payload.email.strip().lower()
    user = db.get_user_by_email(email)
    if user is None or not auth_module.verify_password(payload.password, user["salt"], user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = auth_module.create_token(user["id"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"]}}


@app.get("/api/auth/me")
def me(authorization: str | None = Header(default=None)) -> dict:
    """Return the current user (requires a valid token)."""
    user = current_user_or_none(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"user": {"id": user["id"], "email": user["email"]}}


@app.get("/api/quota")
def quota(
    authorization: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> dict:
    """Report remaining free checks (or unlimited when signed in)."""
    user = current_user_or_none(authorization)
    if user is not None:
        return {"authenticated": True, "used": None, "limit": None, "remaining": None}

    cid = client_id_or_default(x_client_id)
    used = db.get_anon_used(cid)
    return {
        "authenticated": False,
        "used": used,
        "limit": FREE_UPLOADS,
        "remaining": max(0, FREE_UPLOADS - used),
    }


# --------------------------------------------------------------------------
# Video API
# --------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> dict:
    """Accept a video file, store it under uploads/, and register a job.

    Anonymous clients get ``FREE_UPLOADS`` free uploads; afterwards a
    signed-in account is required.
    """
    filename = file.filename or "video.mp4"
    if not is_allowed_file(filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{Path(filename).suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    user = current_user_or_none(authorization)
    cid = client_id_or_default(x_client_id)

    if user is None and db.get_anon_used(cid) >= FREE_UPLOADS:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "login_required",
                "message": f"You've used all {FREE_UPLOADS} free checks. Sign in to continue.",
                "free_limit": FREE_UPLOADS,
            },
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

    # Only count successful uploads against the free quota.
    if user is None:
        used_now = db.increment_anon_used(cid)
    else:
        used_now = None

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
            "user_id": user["id"] if user else None,
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
        "used_free": used_now,
        "free_limit": None if user else FREE_UPLOADS,
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
# Health
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


# --------------------------------------------------------------------------
# Serve the static frontend (mounted last so /api/* routes take precedence)
# --------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
