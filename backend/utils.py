"""Small helper utilities shared across the backend.

This module keeps every "glue" function in one place so the rest of the
application stays focused on its specific concern (video I/O, inference,
HTTP handling).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

#: File extensions the upload endpoint accepts. MP4 is the primary target.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def ensure_dir(path: Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_video_id() -> str:
    """Return a short unique identifier used as the internal key for a job."""
    return uuid.uuid4().hex[:12]


def is_allowed_file(filename: str) -> bool:
    """Return True if the file name has a supported video extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def safe_extension(filename: str) -> str:
    """Return the lowercase extension if supported, else fall back to .mp4."""
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else ".mp4"


@contextmanager
def timing(label: str = ""):
    """Context manager that logs how long the wrapped block took."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Timing[%s]: %.3fs", label or "task", elapsed)


def seconds_to_hms(seconds: float) -> str:
    """Format a number of seconds as a human readable 'm ss.s' string."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"
