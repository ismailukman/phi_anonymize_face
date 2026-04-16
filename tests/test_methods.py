"""Tests for anonymization methods."""

import numpy as np

from phi_anonymize_face.methods import blackbox, blur, pixelate
from phi_anonymize_face.result import BoundingBox


def _make_image():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


def test_blur_modifies_region():
    img = _make_image()
    original = img.copy()
    box = BoundingBox(10, 10, 30, 30)
    blur(img, box, strength=15)
    # Region should be modified
    assert not np.array_equal(img[10:40, 10:40], original[10:40, 10:40])
    # Outside region should be unchanged
    assert np.array_equal(img[0:5, 0:5], original[0:5, 0:5])


def test_pixelate_modifies_region():
    img = _make_image()
    # Add some detail so pixelation is visible
    img[15:25, 15:25] = 255
    original = img.copy()
    box = BoundingBox(10, 10, 30, 30)
    pixelate(img, box, strength=5)
    assert not np.array_equal(img[10:40, 10:40], original[10:40, 10:40])


def test_blackbox_fills_black():
    img = _make_image()
    box = BoundingBox(10, 10, 30, 30)
    blackbox(img, box)
    assert np.all(img[10:40, 10:40] == 0)


def test_empty_box():
    img = _make_image()
    original = img.copy()
    box = BoundingBox(50, 50, 0, 0)
    blur(img, box)
    assert np.array_equal(img, original)
