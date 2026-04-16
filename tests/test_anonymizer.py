"""Tests for the FaceAnonymizer class."""

from pathlib import Path

import numpy as np
import pytest

from phi_anonymize_face import FaceAnonymizer, anonymize_image, anonymize_folder


class TestFaceAnonymizer:
    def test_process_returns_result(self, sample_face_image: Path, tmp_path: Path):
        out = tmp_path / "out.jpg"
        anon = FaceAnonymizer(method="blur", detector="opencv_dnn", fallback=False)
        result = anon.process(sample_face_image, output_path=str(out))
        assert result.success
        assert result.image is not None
        assert result.image.shape[0] > 0

    def test_process_numpy_input(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        anon = FaceAnonymizer(detector="opencv_dnn", fallback=False)
        result = anon.process(img)
        assert result.success
        assert result.image is not None

    def test_process_no_face(self, no_face_image: Path):
        anon = FaceAnonymizer(detector="opencv_dnn", fallback=False)
        result = anon.process(no_face_image)
        assert result.success
        assert result.faces_detected == 0

    def test_process_corrupt(self, corrupt_file: Path):
        anon = FaceAnonymizer(detector="opencv_dnn")
        result = anon.process(corrupt_file)
        assert not result.success
        assert result.error is not None

    def test_process_folder(self, sample_folder: Path, tmp_path: Path):
        out_dir = tmp_path / "output"
        anon = FaceAnonymizer(detector="opencv_dnn", fallback=False)
        results = anon.process_folder(sample_folder, out_dir)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_pixelate_method(self, sample_face_image: Path, tmp_path: Path):
        out = tmp_path / "pix.png"
        anon = FaceAnonymizer(method="pixelate", detector="opencv_dnn", fallback=False)
        result = anon.process(sample_face_image, output_path=str(out))
        assert result.success

    def test_blackbox_method(self, sample_face_image: Path, tmp_path: Path):
        out = tmp_path / "bb.png"
        anon = FaceAnonymizer(method="blackbox", detector="opencv_dnn", fallback=False)
        result = anon.process(sample_face_image, output_path=str(out))
        assert result.success

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            FaceAnonymizer(method="invalid")

    def test_audit_log(self, sample_face_image: Path, tmp_path: Path):
        log = tmp_path / "audit.csv"
        anon = FaceAnonymizer(
            detector="opencv_dnn", fallback=False, audit_log=str(log)
        )
        anon.process(sample_face_image)
        assert log.exists()
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 entry


class TestQuickFunctions:
    def test_anonymize_image(self, sample_face_image: Path, tmp_path: Path):
        out = tmp_path / "quick.jpg"
        result = anonymize_image(
            sample_face_image, output_path=str(out), detector="opencv_dnn", fallback=False
        )
        assert result.success

    def test_anonymize_folder(self, sample_folder: Path, tmp_path: Path):
        out_dir = tmp_path / "quick_out"
        results = anonymize_folder(
            sample_folder, out_dir, detector="opencv_dnn", fallback=False
        )
        assert len(results) == 3
