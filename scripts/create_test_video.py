"""Creates a small synthetic MP4 test video.

Used only to smoke-test the upload/analyze pipeline. It contains moving
patterns, NOT a real human face, so the model output is not meaningful for
real deepfake evaluation - it just verifies the whole system works end to end.

    python scripts/create_test_video.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.video_processor import open_video  # noqa: E402  (used for verification)


def main(out_path: str = "uploads/test_sample.mp4", seconds: float = 3.0, fps: float = 12.0) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    n_frames = int(seconds * fps)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = i / fps
        # Moving colored box + timer text (no face, just pipeline testing).
        x = int((width - 40) * ((i % n_frames) / n_frames))
        y = height // 2
        cv2.rectangle(frame, (x, y - 20), (x + 40, y + 20), (70, 130, 255), -1)
        cv2.putText(frame, f"{t:.1f}s", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()

    cap = open_video(out_path)
    ok = cap.read()[0]
    cap.release()
    if not ok:
        raise SystemExit("Failed to verify generated video.")

    print(f"Created test video: {out_path} ({n_frames} frames, {fps} fps, {seconds}s)")
    return out_path


if __name__ == "__main__":
    main()
