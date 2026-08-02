"""Downloads the pre-trained deepfake detection weights.

Standalone helper so the model can be fetched before the server starts
(used by the Dockerfile and during local setup):
    python scripts/download_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.model import ensure_model_file  # noqa: E402


if __name__ == "__main__":
    path = ensure_model_file()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Model weights ready at {path} ({size_mb:.1f} MB)")
