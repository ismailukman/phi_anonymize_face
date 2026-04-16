"""Core FaceAnonymizer class — the main entry point for the library."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .audit import AuditLogger
from .detectors import CASCADE_ORDER, get_detector
from .io_handler import load_image, save_image
from .methods import METHODS
from .result import AnonymizationResult
from .utils import collect_image_paths, resize_for_detection, scale_boxes

logger = logging.getLogger("phi_anonymize_face")


class FaceAnonymizer:
    """Detect and de-identify faces in images.

    Args:
        method: Anonymization method ("blur", "pixelate", "blackbox").
        blur_strength: Kernel size for blur / block size for pixelate.
        padding: Bounding-box expansion factor (1.3 = 30 % larger).
        detector: Detector backend ("mediapipe", "opencv_dnn", "retinaface", "auto").
        confidence_threshold: Minimum detection confidence.
        fallback: If True and detector finds 0 faces, try next in cascade.
        strip_exif: Strip EXIF metadata from output (default True for PHI safety).
        audit_log: Optional path for CSV audit trail.
    """

    def __init__(
        self,
        method: str = "blur",
        blur_strength: int = 99,
        padding: float = 1.3,
        detector: str = "mediapipe",
        confidence_threshold: float = 0.5,
        fallback: bool = True,
        strip_exif: bool = True,
        audit_log: Optional[str] = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS)}")
        self.method = method
        self.blur_strength = blur_strength
        self.padding = padding
        self.detector_name = detector
        self.confidence_threshold = confidence_threshold
        self.fallback = fallback
        self.strip_exif = strip_exif
        self._audit = AuditLogger(audit_log)
        self._apply = METHODS[method]

    # ---- public API --------------------------------------------------------

    def process(
        self,
        source: Union[str, Path, np.ndarray],
        output_path: Optional[str] = None,
    ) -> AnonymizationResult:
        """Anonymize a single image.

        Args:
            source: Image path or numpy array (BGR).
            output_path: Where to save the result (optional).

        Returns:
            AnonymizationResult with the processed image and metadata.
        """
        src_str = str(source) if not isinstance(source, np.ndarray) else None
        try:
            image = load_image(source)
        except Exception as exc:
            return AnonymizationResult(
                success=False, error=str(exc), source_path=src_str
            )

        boxes = self._detect(image)
        h, w = image.shape[:2]
        padded = [b.pad(self.padding, w, h) for b in boxes]

        out = image.copy()
        for box in padded:
            out = self._apply(out, box, strength=self.blur_strength)

        saved = None
        if output_path:
            saved = save_image(out, output_path, strip_exif=self.strip_exif)

        result = AnonymizationResult(
            image=out,
            faces_detected=len(boxes),
            bounding_boxes=padded,
            output_path=saved,
            success=True,
            source_path=src_str,
        )
        self._audit.log(result, self.method, self.detector_name)
        return result

    def process_folder(
        self,
        folder: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = False,
        max_workers: int = 4,
    ) -> list[AnonymizationResult]:
        """Anonymize all images in a folder.

        Args:
            folder: Input folder path.
            output_dir: Output folder path.
            recursive: Walk subdirectories.
            max_workers: Parallel workers.

        Returns:
            List of AnonymizationResult, one per image.
        """
        folder, output_dir = Path(folder), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = collect_image_paths(folder, recursive=recursive)

        if not paths:
            logger.warning("No supported images found in %s", folder)
            return []

        results: list[AnonymizationResult] = []

        try:
            from tqdm import tqdm

            paths_iter = tqdm(paths, desc="Anonymizing", unit="img")
        except ImportError:
            paths_iter = paths

        def _do(p: Path) -> AnonymizationResult:
            rel = p.relative_to(folder)
            out_p = output_dir / rel
            return self.process(p, output_path=str(out_p))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, p): p for p in paths}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if hasattr(paths_iter, "update"):
                    paths_iter.update(1)  # type: ignore[union-attr]

        return results

    def verify(self, source: Union[str, Path, np.ndarray]) -> bool:
        """Check that an image contains no detectable faces.

        Useful as a post-anonymization validation step.
        """
        try:
            image = load_image(source)
        except Exception:
            return False
        boxes = self._detect(image)
        return len(boxes) == 0

    # ---- internal ----------------------------------------------------------

    def _detect(self, image: np.ndarray) -> list:
        """Run detection, optionally cascading through detectors."""
        resized, scale = resize_for_detection(image)

        if self.detector_name == "auto":
            order = CASCADE_ORDER
        else:
            order = [self.detector_name]
            if self.fallback:
                order += [d for d in CASCADE_ORDER if d != self.detector_name]

        for name in order:
            det = get_detector(name)
            if not det.is_available():
                continue
            boxes = det.detect(resized, self.confidence_threshold)
            if boxes:
                return scale_boxes(boxes, scale)
            if not self.fallback:
                break

        return []
