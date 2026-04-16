"""Black-box redaction anonymization method."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..result import BoundingBox


def blackbox(
    image: np.ndarray,
    box: BoundingBox,
    strength: int = 0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fill the face region with solid black.

    Args:
        image: BGR image (modified in-place).
        box: Padded bounding box.
        strength: Unused (kept for API consistency).
        mask: Optional binary mask (255=face). If provided, only
              masked pixels are blacked out.

    Returns:
        The modified image.
    """
    if mask is not None:
        roi_mask = mask[box.y : box.y2, box.x : box.x2]
        if roi_mask.size > 0:
            image[box.y : box.y2, box.x : box.x2][
                roi_mask > 0
            ] = 0
            return image

    image[box.y : box.y2, box.x : box.x2] = 0
    return image
