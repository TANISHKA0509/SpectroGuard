"""Pre-trained model loading and configuration.

SpectroGuard uses the publicly available **FaceForge Detector** checkpoint
(``huzaifanasirrr/faceforge-detector``) -- an XceptionNet (timm) backbone with
a binary Real/Fake classification head, trained on FaceForensics++ (c40).

The model is **inference only**: weights are downloaded once from the Hugging
Face Hub into ``models/`` and loaded at application startup. No training code
is included by design.

Pre-processing constants below MUST match the model's training setup:
* input size 224x224
* normalise with mean/std = 0.5 (i.e. scale to [-1, 1])
* class index 0 -> REAL, index 1 -> FAKE
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Model registry / download
# --------------------------------------------------------------------------
MODEL_REPO_ID = "huzaifanasirrr/faceforge-detector"
MODEL_FILENAME = "detector_best.pth"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# --------------------------------------------------------------------------
# Pre-processing (must match training)
# --------------------------------------------------------------------------
INPUT_SIZE = 224
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Class index 0 = REAL, 1 = FAKE (as reported by the model author).
CLASS_LABELS = ["REAL", "FAKE"]


def ensure_model_file(force: bool = False) -> Path:
    """Download the pre-trained weights into ``models/`` if missing.

    Uses ``huggingface_hub`` (resumable, cached) with a plain HTTP fallback.
    """
    if MODEL_PATH.exists() and not force:
        logger.info("Model weights already present at %s", MODEL_PATH)
        return MODEL_PATH

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model weights from %s ...", MODEL_REPO_ID)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        shutil.copyfile(cached, MODEL_PATH)
    except Exception as exc:  # fall back to a direct download
        logger.warning("huggingface_hub download failed (%s); falling back to direct URL", exc)
        _download_direct()
    logger.info("Model weights saved to %s", MODEL_PATH)
    return MODEL_PATH


def _download_direct() -> None:
    """Fallback downloader using only the standard library."""
    import urllib.request

    url = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/{MODEL_FILENAME}?download=true"
    tmp = MODEL_PATH.with_suffix(".pth.part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(MODEL_PATH)
    finally:
        tmp.unlink(missing_ok=True)


class DeepfakeDetector(nn.Module):
    """XceptionNet backbone (timm) + the checkpoint's binary classification head.

    The module structure mirrors the checkpoint keys exactly:
    ``xception.<backbone>.*`` and ``classifier.<index>.*``.
    """

    def __init__(self) -> None:
        super().__init__()
        import timm

        # num_classes=0 drops timm's default classifier -> raw 2048-d features.
        backbone = timm.create_model("xception", pretrained=False, num_classes=0)
        head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_LABELS)),
        )
        self.xception = backbone
        self.classifier = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.xception(x))


def load_model(
    weights_path: Path | str = MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[DeepfakeDetector, torch.device]:
    """Load the pre-trained detector into memory and switch to eval mode.

    Returns ``(model, device)``. Called once during application startup.
    """
    ensure_model_file()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeDetector()
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded pre-trained deepfake detector (%d parameters) on %s",
        n_params, device,
    )
    return model, device
