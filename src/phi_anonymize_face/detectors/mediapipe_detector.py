"""MediaPipe-based face detector (primary/fast)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..result import BoundingBox
from .base import BaseDetector


class MediaPipeDetector(BaseDetector):
    """Face detection using MediaPipe Face Detection (Tasks API)."""

    name = "mediapipe"

    def __init__(self) -> None:
        self._detector = None

    def _get_model_path(self) -> str:
        """Locate or download the MediaPipe face detection model."""
        import mediapipe as mp

        # The model ships inside the mediapipe package
        mp_dir = Path(mp.__file__).parent
        candidates = [
            mp_dir / "modules" / "face_detection" / "face_detection_short_range.tflite",
            mp_dir / "models" / "face_detection_short_range.tflite",
        ]

        # Also check env var
        env = os.environ.get("MEDIAPIPE_FACE_MODEL")
        if env:
            candidates.insert(0, Path(env))

        for p in candidates:
            if p.exists():
                return str(p)

        # Download the model
        import urllib.request

        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        cache_dir = Path.home() / ".cache" / "phi_anonymize_face"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "blaze_face_short_range.tflite"
        if not model_path.exists():
            urllib.request.urlretrieve(url, str(model_path))
        return str(model_path)

    def _init_detector(self):
        from mediapipe.tasks.python import vision

        model_path = self._get_model_path()

        from mediapipe.tasks.python import BaseOptions

        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.2,  # we filter later
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[BoundingBox]:
        if self._detector is None:
            self._init_detector()

        import mediapipe as mp

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        boxes: list[BoundingBox] = []
        if not result.detections:
            return boxes

        h, w = image.shape[:2]
        for det in result.detections:
            score = det.categories[0].score if det.categories else 0.0
            if score < confidence_threshold:
                continue
            bb = det.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            bw = min(bb.width, w - x)
            bh = min(bb.height, h - y)
            if bw > 0 and bh > 0:
                boxes.append(BoundingBox(x, y, bw, bh, score))
        return boxes

    def is_available(self) -> bool:
        try:
            import mediapipe  # noqa: F401

            return True
        except ImportError:
            return False
