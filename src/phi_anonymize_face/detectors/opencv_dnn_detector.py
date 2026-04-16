"""OpenCV DNN face detector (SSD-based fallback)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..result import BoundingBox
from .base import BaseDetector

# OpenCV ships a pre-trained Caffe face-detection model with the contrib/dnn module.
# We use the Yunet model that ships with OpenCV 4.8+.
_YUNET_AVAILABLE = hasattr(cv2, "FaceDetectorYN_create") or hasattr(
    cv2, "FaceDetectorYN"
)


class OpenCVDNNDetector(BaseDetector):
    """Face detection using OpenCV DNN (YuNet or SSD Caffe fallback)."""

    name = "opencv_dnn"

    def __init__(self) -> None:
        self._detector = None

    def _init_detector(self, w: int, h: int):
        """Initialize the YuNet face detector (ships with OpenCV 4.8+)."""
        model_path = self._find_yunet_model()
        if model_path and hasattr(cv2, "FaceDetectorYN"):
            self._detector = cv2.FaceDetectorYN.create(
                model_path, "", (w, h), 0.5, 0.3, 5000
            )
            self._backend = "yunet"
        else:
            # Fallback to Haar cascades (always available)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)
            self._backend = "haar"

    @staticmethod
    def _find_yunet_model() -> str | None:
        """Try to locate the YuNet ONNX model."""
        candidates = [
            Path(cv2.data.haarcascades).parent / "face_detection_yunet_2023mar.onnx",
            Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx",
        ]
        env = os.environ.get("YUNET_MODEL_PATH")
        if env:
            candidates.insert(0, Path(env))

        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[BoundingBox]:
        h, w = image.shape[:2]
        if self._detector is None:
            self._init_detector(w, h)

        boxes: list[BoundingBox] = []

        if self._backend == "yunet":
            self._detector.setInputSize((w, h))
            self._detector.setScoreThreshold(confidence_threshold)
            _, faces = self._detector.detect(image)
            if faces is not None:
                for face in faces:
                    x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    score = float(face[14]) if face.shape[0] > 14 else float(face[-1])
                    if score >= confidence_threshold and fw > 0 and fh > 0:
                        x = max(0, x)
                        y = max(0, y)
                        boxes.append(BoundingBox(x, y, fw, fh, score))
        else:
            # Haar cascade fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, fw, fh in rects:
                boxes.append(
                    BoundingBox(int(x), int(y), int(fw), int(fh), confidence=0.9)
                )

        return boxes
