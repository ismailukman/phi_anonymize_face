"""Shared test fixtures — generates synthetic face images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def sample_face_image(tmp_path: Path) -> Path:
    """Create a synthetic image with a detectable face-like pattern.

    Uses OpenCV's Haar cascade drawing heuristic: an oval + eyes + mouth
    in roughly the right proportions triggers most detectors.
    """
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # light gray bg

    # Head oval
    cv2.ellipse(img, (320, 200), (90, 120), 0, 0, 360, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (290, 175), 12, (40, 40, 40), -1)
    cv2.circle(img, (350, 175), 12, (40, 40, 40), -1)
    # Mouth
    cv2.ellipse(img, (320, 240), (30, 12), 0, 0, 180, (80, 50, 50), 2)

    path = tmp_path / "face.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def no_face_image(tmp_path: Path) -> Path:
    """Create a plain image with no face."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
    path = tmp_path / "noface.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def sample_folder(tmp_path: Path, sample_face_image: Path) -> Path:
    """Create a folder with multiple images."""
    folder = tmp_path / "images"
    folder.mkdir()
    import shutil

    for i in range(3):
        shutil.copy(sample_face_image, folder / f"img_{i}.jpg")
    return folder


@pytest.fixture()
def corrupt_file(tmp_path: Path) -> Path:
    """Create a corrupt image file."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image")
    return path
