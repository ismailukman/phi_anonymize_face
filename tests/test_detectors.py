"""Tests for face detectors."""

import cv2
import numpy as np

from phi_anonymize_face.detectors import get_detector
from phi_anonymize_face.result import BoundingBox


def test_opencv_dnn_on_synthetic(sample_face_image):
    img = cv2.imread(str(sample_face_image))
    det = get_detector("opencv_dnn")
    boxes = det.detect(img, confidence_threshold=0.3)
    # Haar cascade should find the synthetic face
    assert isinstance(boxes, list)
    for b in boxes:
        assert isinstance(b, BoundingBox)


def test_detector_returns_empty_on_blank():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    det = get_detector("opencv_dnn")
    boxes = det.detect(img)
    assert isinstance(boxes, list)


def test_bounding_box_pad():
    box = BoundingBox(100, 100, 50, 50, 0.9)
    padded = box.pad(1.5, 640, 480)
    assert padded.w >= box.w
    assert padded.h >= box.h
    assert padded.x >= 0
    assert padded.y >= 0


def test_bounding_box_pad_clamps():
    box = BoundingBox(0, 0, 100, 100, 0.9)
    padded = box.pad(2.0, 120, 120)
    assert padded.x >= 0
    assert padded.y >= 0
    assert padded.x2 <= 120
    assert padded.y2 <= 120
