"""Gaussian blur anonymization method."""

from __future__ import annotations

import cv2
import numpy as np

from ..result import BoundingBox


def blur(
    image: np.ndarray,
    box: BoundingBox,
    strength: int = 99,
) -> np.ndarray:
    """Apply Gaussian blur to the face region.

    Args:
        image: BGR image (modified in-place).
        box: Padded bounding box of the face.
        strength: Kernel size (must be odd; will be made odd if even).

    Returns:
        The modified image.
    """
    k = strength if strength % 2 == 1 else strength + 1
    roi = image[box.y : box.y2, box.x : box.x2]
    if roi.size == 0:
        return image
    image[box.y : box.y2, box.x : box.x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return image
