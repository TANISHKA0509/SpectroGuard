# SpectroGuard - container image
# Used for local Docker and cloud deployment (Render / Hugging Face Spaces).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# CPU-only PyTorch keeps the image small (~800 MB vs multi-GB CUDA images).
# The CPU index has x86_64 wheels; on ARM64 (e.g. Oracle free VMs) fall back to
# PyPI's CPU-only aarch64 wheels.
COPY requirements.txt ./
RUN (pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu \
     || pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1) \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch model weights at build time so the container starts fast.
RUN python scripts/download_model.py

# Render passes a $PORT; Hugging Face Spaces expects 7860. Default: 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
