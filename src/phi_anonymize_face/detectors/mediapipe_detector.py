"""MediaPipe-based face detector (primary/fast)."""

from __future__ import annotations

import numpy as np

from ..result import BoundingBox
from .base import BaseDetector


class MediaPipeDetector(BaseDetector):
    """Face detection using MediaPipe Face Detection."""

    name = "mediapipe"

    def __init__(self, model_selection: int = 1) -> None:
        """Init.

        Args:
            model_selection: 0 for short-range (< 2 m), 1 for full-range (< 5 m).
        """
        self._model_selection = model_selection
        self._detector = None

    def _init_detector(self):
        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=0.2,  # we filter later
        )

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[BoundingBox]:
        if self._detector is None:
            self._init_detector()

        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        boxes: list[BoundingBox] = []
        if not results.detections:
            return boxes

        h, w = image.shape[:2]
        for det in results.detections:
            score = det.score[0]
            if score < confidence_threshold:
                continue
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width * w), w - x)
            bh = min(int(bb.height * h), h - y)
            if bw > 0 and bh > 0:
                boxes.append(BoundingBox(x, y, bw, bh, score))
        return boxes

    def is_available(self) -> bool:
        try:
            import mediapipe  # noqa: F401

            return True
        except ImportError:
            return False
