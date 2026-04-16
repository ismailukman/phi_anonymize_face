"""Pixelation anonymization method."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..result import BoundingBox


def pixelate(
    image: np.ndarray,
    box: BoundingBox,
    strength: int = 10,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pixelate the face region by downscaling and upscaling.

    Args:
        image: BGR image (modified in-place).
        box: Padded bounding box.
        strength: Block size in pixels (smaller = more pixelated).
        mask: Optional binary mask (255=face). If provided, only
              masked pixels are pixelated.

    Returns:
        The modified image.
    """
    roi = image[box.y : box.y2, box.x : box.x2]
    if roi.size == 0:
        return image
    h, w = roi.shape[:2]
    block = max(2, strength)
    small = cv2.resize(
        roi,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    pixelated_roi = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST
    )

    if mask is not None:
        roi_mask = mask[box.y : box.y2, box.x : box.x2]
        if roi_mask.size > 0:
            m = roi_mask[:, :, np.newaxis] / 255.0
            blended = (pixelated_roi * m + roi * (1 - m)).astype(
                np.uint8
            )
            image[box.y : box.y2, box.x : box.x2] = blended
            return image

    image[box.y : box.y2, box.x : box.x2] = pixelated_roi
    return image
