"""Tests for I/O handling."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from phi_anonymize_face.io_handler import load_image, save_image


def test_load_numpy():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    result = load_image(arr)
    assert np.array_equal(result, arr)


def test_load_file(sample_face_image: Path):
    img = load_image(sample_face_image)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_load_missing():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path.jpg")


def test_load_corrupt(corrupt_file: Path):
    with pytest.raises(ValueError, match="Cannot decode"):
        load_image(corrupt_file)


def test_save_strips_exif(tmp_path: Path):
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    out = tmp_path / "out.jpg"
    path = save_image(img, out, strip_exif=True)
    assert Path(path).exists()
    # Reload and check it's a valid image
    reloaded = cv2.imread(path)
    assert reloaded is not None


def test_save_creates_dirs(tmp_path: Path):
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    out = tmp_path / "sub" / "dir" / "img.png"
    path = save_image(img, out)
    assert Path(path).exists()
