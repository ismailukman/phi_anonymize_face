"""Data classes for anonymization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """A face bounding box with confidence score."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def pad(self, factor: float, img_w: int, img_h: int) -> "BoundingBox":
        """Return a new padded bounding box clamped to image bounds."""
        cx, cy = self.x + self.w / 2, self.y + self.h / 2
        nw, nh = self.w * factor, self.h * factor
        nx = max(0, int(cx - nw / 2))
        ny = max(0, int(cy - nh / 2))
        nx2 = min(img_w, int(cx + nw / 2))
        ny2 = min(img_h, int(cy + nh / 2))
        return BoundingBox(nx, ny, nx2 - nx, ny2 - ny, self.confidence)


@dataclass
class AnonymizationResult:
    """Result of processing a single image."""

    image: Optional[np.ndarray] = None
    faces_detected: int = 0
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    output_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    source_path: Optional[str] = None
