"""phi_anonymize_face — HIPAA-compliant face de-identification for medical images."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .anonymizer import FaceAnonymizer
from .result import AnonymizationResult, BoundingBox

__version__ = "0.2.0"


def anonymize_image(
    source: Union[str, Path, np.ndarray],
    method: str = "blur",
    output_path: Optional[str] = None,
    **kwargs,
) -> AnonymizationResult:
    """Quick function to anonymize a single image.

    Args:
        source: Image file path or numpy array.
        method: "blur", "pixelate", or "blackbox".
        output_path: Optional save path.
        **kwargs: Forwarded to FaceAnonymizer.

    Returns:
        AnonymizationResult.
    """
    anon = FaceAnonymizer(method=method, **kwargs)
    return anon.process(source, output_path=output_path)


def anonymize_folder(
    folder: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "blur",
    recursive: bool = False,
    **kwargs,
) -> list[AnonymizationResult]:
    """Quick function to anonymize all images in a folder.

    Args:
        folder: Input folder.
        output_dir: Output folder.
        method: "blur", "pixelate", or "blackbox".
        recursive: Walk subdirectories.
        **kwargs: Forwarded to FaceAnonymizer.

    Returns:
        List of AnonymizationResult.
    """
    anon = FaceAnonymizer(method=method, **kwargs)
    return anon.process_folder(folder, output_dir, recursive=recursive)


__all__ = [
    "FaceAnonymizer",
    "AnonymizationResult",
    "BoundingBox",
    "anonymize_image",
    "anonymize_folder",
    "__version__",
]
