"""Publish SpectroGuard to Hugging Face Spaces (free public URL).

What it does:
  1. Creates (or reuses) a Docker-based Space named `spectroguard` under your
     Hugging Face account.
  2. Uploads the whole project. The Space build then:
       - installs dependencies,
       - downloads the pre-trained model,
       - starts the FastAPI server on port 7860.
  3. You get a live URL like https://<username>-spectroguard.hf.space

Prerequisites:
  * Create a free account at https://huggingface.co/join
  * Authenticate (one of):
        pip install huggingface_hub        (already installed)
        huggingface-cli login               # paste your token
    or set the HF_TOKEN environment variable.

Usage:
    python scripts/deploy_space.py
    python scripts/deploy_space.py --name spectroguard --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parent.parent

# Never upload the venv, local weights, uploads or git internals.
IGNORE = [
    ".venv", "venv", "__pycache__", ".git", ".gitignore",
    "models", "uploads", "*.pth", "*.pt",
    "server.log", "server.err.log",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy SpectroGuard to HF Spaces.")
    parser.add_argument("--name", default="spectroguard", help="Space name (default: spectroguard)")
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    args = parser.parse_args()

    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.name}"

    print(f"[1/2] Creating space {repo_id} (sdk=docker)...")
    create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)

    print(f"[2/2] Uploading project from {ROOT}...")
    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
    )

    print("\nDone! Your live app will be at:")
    print(f"    https://{username}-spectroguard.hf.space")
    print("The Space build (deps + model download) takes a few minutes the first time.")


if __name__ == "__main__":
    main()
