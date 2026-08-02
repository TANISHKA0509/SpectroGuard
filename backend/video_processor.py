"""Frame extraction and pre-processing using OpenCV.

Responsibilities
----------------
* Validate that an uploaded file is a readable video.
* Sample every ``n``-th frame (instead of every frame) to keep inference fast.
* Optionally crop the largest detected face (Haar cascade) before resizing,
  because deepfake models are trained on face regions.
* Convert a BGR OpenCV frame into the normalized tensor the model expects.

The pre-processing steps (resize size, mean/std) must match exactly what the
pre-trained model was trained with -- see ``model.INPUT_SIZE``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from . import model as model_module
from .utils import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

#: Sampling policy - analyse every Nth frame, at most MAX_FRAMES frames.
FRAME_SKIP = 10
MAX_FRAMES = 30

#: Smallest face box (in pixels) we bother cropping.
MIN_FACE_SIZE = 64

_FACE_CASCADE = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Lazily load OpenCV's bundled frontal-face Haar cascade."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


class UnsupportedVideoError(ValueError):
    """Raised when a file cannot be opened/read as a video."""


def is_supported_video(filename: str) -> bool:
    """True if the filename has an allowed video extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def open_video(path: str | Path) -> cv2.VideoCapture:
    """Open a video file and raise if it cannot be decoded."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise UnsupportedVideoError(f"Could not open video file: {path}")
    return cap


def extract_frames(
    video_path: str | Path,
    frame_skip: int = FRAME_SKIP,
    max_frames: int = MAX_FRAMES,
) -> tuple[list[np.ndarray], dict]:
    """Extract sampled frames (BGR) from a video.

    Iterates the whole file but only keeps every ``frame_skip``-th frame,
    stopping after ``max_frames`` samples. Returns the list of frames plus a
    metadata dictionary (total frame count, fps, how many were sampled).
    """
    cap = open_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_skip == 0:
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()

    if not frames:
        raise UnsupportedVideoError("No readable frames found in the video.")

    metadata = {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frames_sampled": len(frames),
        "sampling_ratio": f"1 in {frame_skip}",
    }
    logger.info(
        "Extracted %d frames from %s (skipping every %dth)",
        len(frames), video_path, frame_skip,
    )
    return frames, metadata


def crop_face(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Crop the largest face region; fall back to the full frame if none found.

    Returns ``(image, face_found)`` where ``image`` is the crop (or the
    original frame when no face was detected).
    """
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    if len(faces) == 0:
        return frame, False

    # Keep the largest face box (most likely the subject of a video call).
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Add a margin around the box so the crop contains the forehead/chin.
    mx, my = int(0.2 * w), int(0.2 * h)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame, False
    return crop, True


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Convert a BGR frame into a ``1 x 3 x H x W`` normalized float tensor.

    Pipeline: BGR -> RGB -> resize to model input -> scale to [0,1] ->
    normalize with the model's mean/std -> CHW tensor.
    """
    size = model_module.INPUT_SIZE
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    arr = resized.astype(np.float32) / 255.0
    mean = np.asarray(model_module.NORMALIZE_MEAN, dtype=np.float32)
    std = np.asarray(model_module.NORMALIZE_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

    return torch.from_numpy(arr).unsqueeze(0)
