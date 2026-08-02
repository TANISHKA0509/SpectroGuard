"""Prediction pipeline: run the pre-trained model over sampled frames and
aggregate per-frame predictions into a single video-level verdict.

Frame predictions are combined with **majority voting** and the final
confidence is the *average probability the model assigned to the winning
class* across all sampled frames.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from . import model as model_module
from .video_processor import crop_face, extract_frames, preprocess_frame

logger = logging.getLogger(__name__)


def build_batch(
    frames: list[np.ndarray],
    use_face_crop: bool = True,
) -> tuple[torch.Tensor, int]:
    """Preprocess a list of BGR frames into one batched input tensor.

    Returns ``(batch_tensor, face_found_count)``.
    """
    tensors = []
    face_found = 0
    for frame in frames:
        if use_face_crop:
            frame, found = crop_face(frame)
            if found:
                face_found += 1
        tensors.append(preprocess_frame(frame))
    return torch.cat(tensors, dim=0), face_found


def run_inference(
    model: torch.nn.Module,
    device: torch.device,
    video_path: str,
    frame_skip: int = 10,
    max_frames: int = 30,
) -> dict:
    """Run the full frame-level inference on a video file.

    Returns a dictionary containing the per-frame probabilities plus
    metadata used by the API layer to build the final response.
    """
    start = time.perf_counter()

    frames, meta = extract_frames(video_path, frame_skip, max_frames)
    batch, face_found = build_batch(frames, use_face_crop=True)

    with torch.no_grad():
        logits = model(batch.to(device))
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 2)

    frame_predictions = probabilities.argmax(axis=1)
    frame_confidences = probabilities[np.arange(len(probabilities)), frame_predictions]

    elapsed = time.perf_counter() - start
    logger.info(
        "Inference on %d frames took %.2fs (face found in %d)",
        len(frames), elapsed, face_found,
    )

    return {
        "probabilities": probabilities,
        "frame_predictions": frame_predictions,
        "frame_confidences": frame_confidences,
        "metadata": meta,
        "face_found": face_found,
        "inference_time": round(elapsed, 2),
    }


def aggregate_video_prediction(
    probabilities: np.ndarray,
    frame_predictions: np.ndarray,
    frame_confidences: np.ndarray,
) -> dict:
    """Combine per-frame outputs into the final video-level prediction.

    * ``prediction``: majority vote over per-frame class indices.
    * ``confidence``: mean probability of the winning class across frames.
    * ``votes``: how many frames voted for each class.
    """
    n = len(frame_predictions)
    if n == 0:
        raise ValueError("No frame predictions to aggregate.")

    votes_real = int(np.sum(frame_predictions == 0))
    votes_fake = int(np.sum(frame_predictions == 1))

    final_idx = 1 if votes_fake > votes_real else 0  # ties -> real
    final_label = model_module.CLASS_LABELS[final_idx]

    # Average confidence in the winning class across all sampled frames.
    confidence = float(np.mean(probabilities[:, final_idx]))
    avg_real = float(np.mean(probabilities[:, 0]))
    avg_fake = float(np.mean(probabilities[:, 1]))

    return {
        "prediction": final_label,
        "confidence": round(confidence, 4),
        "frames_analyzed": n,
        "votes": {
            "REAL": votes_real,
            "FAKE": votes_fake,
        },
        "average_confidence_real": round(avg_real, 4),
        "average_confidence_fake": round(avg_fake, 4),
    }
