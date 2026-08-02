# Test runner for SpectroGuard
# Verifies the full API pipeline against a live server (default http://127.0.0.1:8001)
#
# Usage:
#   python scripts/smoke_test.py            # uses uploads/test_sample.mp4 if present
#   python scripts/smoke_test.py my_video.mp4
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8001"
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "uploads/test_sample.mp4"


def main() -> None:
    video_path = Path(VIDEO)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    r = requests.get(f"{BASE_URL}/health", timeout=10)
    r.raise_for_status()
    print("health:", r.json())

    with video_path.open("rb") as fh:
        r = requests.post(f"{BASE_URL}/api/upload", files={"file": fh}, timeout=60)
    r.raise_for_status()
    video_id = r.json()["video_id"]
    print("upload:", r.json())

    r = requests.post(f"{BASE_URL}/api/analyze/{video_id}", timeout=10)
    r.raise_for_status()
    print("analyze:", r.json())

    while True:
        r = requests.get(f"{BASE_URL}/api/status/{video_id}", timeout=10)
        r.raise_for_status()
        status = r.json()
        print("status:", status["status"], status["progress"], status.get("message", ""))
        if status["status"] in ("done", "error"):
            break
        time.sleep(1)

    if status["status"] == "error":
        raise SystemExit(f"Analysis failed: {status.get('error')}")

    r = requests.get(f"{BASE_URL}/api/result/{video_id}", timeout=10)
    r.raise_for_status()
    result = r.json()["result"]
    print("\n===== RESULT =====")
    print(f"Prediction: {result['prediction']}  (confidence {result['confidence']:.2%})")
    print(f"Frames: {result['frames_analyzed']}  Votes: {result['votes']}  Faces: {result['faces_detected']}")
    print(f"Processing time: {result['processing_time']}s")


if __name__ == "__main__":
    main()
