"""Face detector registry."""

from __future__ import annotations

from .base import BaseDetector
from .mediapipe_detector import MediaPipeDetector
from .opencv_dnn_detector import OpenCVDNNDetector
from .retinaface_detector import RetinaFaceDetector

DETECTORS: dict[str, type[BaseDetector]] = {
    "mediapipe": MediaPipeDetector,
    "opencv_dnn": OpenCVDNNDetector,
    "retinaface": RetinaFaceDetector,
}

# Cascade order for "auto" mode
CASCADE_ORDER = ["mediapipe", "opencv_dnn", "retinaface"]


def get_detector(name: str) -> BaseDetector:
    """Instantiate a detector by name."""
    if name not in DETECTORS:
        raise ValueError(
            f"Unknown detector '{name}'. Choose from: {list(DETECTORS.keys())}"
        )
    return DETECTORS[name]()


__all__ = [
    "BaseDetector",
    "MediaPipeDetector",
    "OpenCVDNNDetector",
    "RetinaFaceDetector",
    "DETECTORS",
    "CASCADE_ORDER",
    "get_detector",
]
