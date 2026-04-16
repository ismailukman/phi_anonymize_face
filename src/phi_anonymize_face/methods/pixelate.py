"""Pixelation anonymization method."""

from __future__ import annotations

import cv2
import numpy as np

from ..result import BoundingBox


def pixelate(
    image: np.ndarray,
    box: BoundingBox,
    strength: int = 10,
) -> np.ndarray:
    """Pixelate the face region by downscaling and upscaling.

    Args:
        image: BGR image (modified in-place).
        box: Padded bounding box.
        strength: Block size in pixels (smaller = more pixelated).

    Returns:
        The modified image.
    """
    roi = image[box.y : box.y2, box.x : box.x2]
    if roi.size == 0:
        return image
    h, w = roi.shape[:2]
    block = max(2, strength)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    image[box.y : box.y2, box.x : box.x2] = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST
    )
    return image
