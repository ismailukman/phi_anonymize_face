"""Audit logging for PHI compliance tracking."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .result import AnonymizationResult

logger = logging.getLogger("phi_anonymize_face")


class AuditLogger:
    """Logs every anonymization action to a CSV audit trail."""

    FIELDS = [
        "timestamp",
        "source_path",
        "output_path",
        "faces_detected",
        "method",
        "detector",
        "success",
        "error",
    ]

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                with open(self.log_path, "w", newline="") as f:
                    csv.writer(f).writerow(self.FIELDS)

    def log(
        self,
        result: AnonymizationResult,
        method: str,
        detector: str,
    ) -> None:
        """Record a processing event."""
        ts = datetime.now(timezone.utc).isoformat()
        row = [
            ts,
            result.source_path or "",
            result.output_path or "",
            result.faces_detected,
            method,
            detector,
            result.success,
            result.error or "",
        ]
        logger.info(
            "Processed %s — %d faces, method=%s, success=%s",
            result.source_path,
            result.faces_detected,
            method,
            result.success,
        )
        if self.log_path:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
