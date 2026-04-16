"""Gaussian blur anonymization method."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..result import BoundingBox


def blur(
    image: np.ndarray,
    box: BoundingBox,
    strength: int = 99,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply Gaussian blur to the face region.

    Args:
        image: BGR image (modified in-place).
        box: Padded bounding box of the face.
        strength: Kernel size (must be odd; will be made odd if even).
        mask: Optional binary mask (255=face). If provided, only
              masked pixels are blurred (precise face mode).

    Returns:
        The modified image.
    """
    k = strength if strength % 2 == 1 else strength + 1
    roi = image[box.y : box.y2, box.x : box.x2]
    if roi.size == 0:
        return image

    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)

    if mask is not None:
        roi_mask = mask[box.y : box.y2, box.x : box.x2]
        if roi_mask.size > 0:
            m = roi_mask[:, :, np.newaxis] / 255.0
            blended = (blurred_roi * m + roi * (1 - m)).astype(
                np.uint8
            )
            image[box.y : box.y2, box.x : box.x2] = blended
            return image

    image[box.y : box.y2, box.x : box.x2] = blurred_roi
    return image
