"""Background worker thread for anonymization tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..anonymizer import FaceAnonymizer
from ..result import AnonymizationResult


class AnonymizeWorker(QThread):
    """Runs anonymization in a background thread so the UI stays responsive."""

    progress = pyqtSignal(int, int)  # (current, total)
    image_done = pyqtSignal(object)  # AnonymizationResult
    finished_all = pyqtSignal(list)  # list[AnonymizationResult]
    error = pyqtSignal(str)

    def __init__(
        self,
        anonymizer: FaceAnonymizer,
        paths: list[Path],
        output_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.anonymizer = anonymizer
        self.paths = paths
        self.output_dir = output_dir
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        results: list[AnonymizationResult] = []
        total = len(self.paths)

        for i, path in enumerate(self.paths):
            if self._cancel:
                break

            out_path = self.output_dir / path.name
            try:
                result = self.anonymizer.process(path, output_path=str(out_path))
            except Exception as exc:
                result = AnonymizationResult(
                    success=False, error=str(exc), source_path=str(path)
                )

            results.append(result)
            self.image_done.emit(result)
            self.progress.emit(i + 1, total)

        self.finished_all.emit(results)
