"""Utility helpers for padding, resizing, and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".dcm"}


def is_image_file(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_image_paths(folder: Path, recursive: bool = False) -> list[Path]:
    """Collect all supported image paths from a folder."""
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and is_image_file(p))


def resize_for_detection(
    image: np.ndarray, max_dim: int = 1920
) -> tuple[np.ndarray, float]:
    """Resize image for faster detection, return (resized, scale_factor).

    The scale factor maps detection-coordinate back to the original.
    """
    import cv2

    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, 1.0
    scale = max_dim / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized, scale


def scale_boxes(
    boxes: list, scale: float
) -> list:
    """Scale bounding boxes back to original image coordinates."""
    from .result import BoundingBox

    if scale == 1.0:
        return boxes
    inv = 1.0 / scale
    return [
        BoundingBox(
            int(b.x * inv), int(b.y * inv),
            int(b.w * inv), int(b.h * inv),
            b.confidence,
        )
        for b in boxes
    ]
