"""Smoke tests for the CLI."""

from pathlib import Path

from click.testing import CliRunner

from phi_anonymize_face.cli import main


def test_cli_single_image(sample_face_image: Path, tmp_path: Path):
    out = tmp_path / "cli_out.jpg"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["-i", str(sample_face_image), "-o", str(out),
         "--detector", "opencv_dnn", "--no-fallback"],
    )
    assert result.exit_code == 0
    assert "Done" in result.output or "0 face" in result.output


def test_cli_folder(sample_folder: Path, tmp_path: Path):
    out_dir = tmp_path / "cli_out_dir"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["-i", str(sample_folder), "-o", str(out_dir),
         "--detector", "opencv_dnn", "--no-fallback"],
    )
    assert result.exit_code == 0
    assert "Processed" in result.output


def test_cli_invalid_input(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["-i", str(tmp_path / "nonexistent.jpg"), "-o", str(tmp_path / "out.jpg")],
    )
    assert result.exit_code != 0
