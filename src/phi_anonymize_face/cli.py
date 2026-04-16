"""Command-line interface for phi_anonymize_face."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .anonymizer import FaceAnonymizer


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="phi-anonymize")
@click.option("-i", "--input", "input_path", required=True, help="Image or folder path.")
@click.option("-o", "--output", "output_path", required=True, help="Output path.")
@click.option(
    "-m", "--method",
    type=click.Choice(["blur", "pixelate", "blackbox"]),
    default="blur",
    show_default=True,
    help="Anonymization method.",
)
@click.option(
    "--blur-strength", default=99, show_default=True,
    help="Blur kernel / pixelate block size.",
)
@click.option(
    "--padding", default=1.3, show_default=True,
    help="Bounding-box padding factor.",
)
@click.option(
    "--detector", default="mediapipe", show_default=True,
    help="Detector: mediapipe, opencv_dnn, retinaface, auto.",
)
@click.option(
    "--confidence", default=0.5, show_default=True,
    help="Min detection confidence.",
)
@click.option("--no-fallback", is_flag=True, help="Disable cascade fallback.")
@click.option("--recursive", is_flag=True, help="Recurse into subdirectories.")
@click.option("--keep-exif", is_flag=True, help="Preserve EXIF metadata.")
@click.option("--audit-log", default=None, help="Path for CSV audit log.")
@click.option("--verify", is_flag=True, help="Verify output images have no residual faces.")
def main(
    input_path: str,
    output_path: str,
    method: str,
    blur_strength: int,
    padding: float,
    detector: str,
    confidence: float,
    no_fallback: bool,
    recursive: bool,
    keep_exif: bool,
    audit_log: str | None,
    verify: bool,
) -> None:
    """Anonymize faces in medical images for HIPAA/PHI compliance."""
    anon = FaceAnonymizer(
        method=method,
        blur_strength=blur_strength,
        padding=padding,
        detector=detector,
        confidence_threshold=confidence,
        fallback=not no_fallback,
        strip_exif=not keep_exif,
        audit_log=audit_log,
    )

    inp = Path(input_path)
    if inp.is_dir():
        results = anon.process_folder(inp, output_path, recursive=recursive)
        total = len(results)
        ok = sum(1 for r in results if r.success)
        faces = sum(r.faces_detected for r in results)
        click.echo(f"Processed {total} images — {faces} faces anonymized, {ok} succeeded.")

        if verify:
            fails = []
            for r in results:
                if r.output_path and not anon.verify(r.output_path):
                    fails.append(r.output_path)
            if fails:
                click.echo(f"VERIFICATION FAILED — {len(fails)} images still contain faces:")
                for f in fails:
                    click.echo(f"  {f}")
                sys.exit(1)
            else:
                click.echo("Verification passed: no residual faces detected.")
    else:
        result = anon.process(inp, output_path=output_path)
        if result.success:
            click.echo(
                f"Done — {result.faces_detected} face(s) anonymized → {result.output_path}"
            )
            if verify and result.output_path:
                if anon.verify(result.output_path):
                    click.echo("Verification passed.")
                else:
                    click.echo("VERIFICATION FAILED — residual face detected.")
                    sys.exit(1)
        else:
            click.echo(f"Error: {result.error}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
