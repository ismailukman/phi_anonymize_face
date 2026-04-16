"""Anonymization method registry."""

from __future__ import annotations

from .blackbox import blackbox
from .blur import blur
from .pixelate import pixelate

METHODS = {
    "blur": blur,
    "pixelate": pixelate,
    "blackbox": blackbox,
}

__all__ = ["METHODS", "blur", "pixelate", "blackbox"]
