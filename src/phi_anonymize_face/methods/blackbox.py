"""Black-box redaction anonymization method."""

from __future__ import annotations

import numpy as np

from ..result import BoundingBox


def blackbox(
    image: np.ndarray,
    box: BoundingBox,
    strength: int = 0,
) -> np.ndarray:
    """Fill the face region with a solid black rectangle.

    Args:
        image: BGR image (modified in-place).
        box: Padded bounding box.
        strength: Unused (kept for API consistency).

    Returns:
        The modified image.
    """
    image[box.y : box.y2, box.x : box.x2] = 0
    return image
