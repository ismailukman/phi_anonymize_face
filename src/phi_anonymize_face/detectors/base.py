"""Abstract base class for face detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..result import BoundingBox


class BaseDetector(ABC):
    """Interface that all face detectors must implement."""

    name: str = "base"

    @abstractmethod
    def detect(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[BoundingBox]:
        """Detect faces in a BGR image.

        Args:
            image: BGR numpy array.
            confidence_threshold: Minimum confidence to keep a detection.

        Returns:
            List of BoundingBox instances.
        """

    def is_available(self) -> bool:
        """Check whether this detector's dependencies are installed."""
        return True
