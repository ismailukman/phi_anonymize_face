"""Image and DICOM I/O with EXIF stripping."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image


def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from path or pass through a numpy array.

    Args:
        source: File path (str/Path) or numpy array (BGR).

    Returns:
        BGR numpy array.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If image cannot be decoded.
    """
    if isinstance(source, np.ndarray):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() == ".dcm":
        return _load_dicom(path)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def _load_dicom(path: Path) -> np.ndarray:
    """Load pixel data from a DICOM file and convert to BGR."""
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install it with: pip install phi-anonymize-face[dicom]"
        )

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array

    # Handle grayscale or RGB
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Normalize to uint8 if necessary
    if arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
            np.uint8
        )
    return arr


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    strip_exif: bool = True,
) -> str:
    """Save image to disk, optionally stripping EXIF metadata.

    Args:
        image: BGR numpy array.
        output_path: Destination path.
        strip_exif: If True, save without EXIF data (default for PHI safety).

    Returns:
        Absolute path string of saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".dcm":
        return _save_dicom(image, output_path)

    if strip_exif:
        # Convert BGR→RGB, save via Pillow (no EXIF)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(str(output_path))
    else:
        cv2.imwrite(str(output_path), image)

    return str(output_path.resolve())


def _save_dicom(image: np.ndarray, output_path: Path) -> str:
    """Save image as a basic DICOM file."""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install it with: pip install phi-anonymize-face[dicom]"
        )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ds = FileDataset(
        str(output_path), {}, preamble=b"\x00" * 128, is_implicit_VR=False
    )
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()

    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.PixelData = rgb.tobytes()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = generate_uid()

    ds.save_as(str(output_path))
    return str(output_path.resolve())
