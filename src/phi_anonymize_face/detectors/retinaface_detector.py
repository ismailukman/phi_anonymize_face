"""RetinaFace detector via insightface (optional, high-recall)."""

from __future__ import annotations

import numpy as np

from ..result import BoundingBox
from .base import BaseDetector


class RetinaFaceDetector(BaseDetector):
    """Face detection using InsightFace RetinaFace model."""

    name = "retinaface"

    def __init__(self) -> None:
        self._app = None

    def _init_detector(self):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_sc", providers=["CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[BoundingBox]:
        if self._app is None:
            self._init_detector()

        faces = self._app.get(image)
        boxes: list[BoundingBox] = []
        for face in faces:
            score = float(face.det_score)
            if score < confidence_threshold:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append(BoundingBox(x1, y1, w, h, score))
        return boxes

    def is_available(self) -> bool:
        try:
            import insightface  # noqa: F401
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False
