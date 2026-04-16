"""Main GUI application window."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..anonymizer import FaceAnonymizer
from ..result import AnonymizationResult
from ..utils import SUPPORTED_EXTENSIONS
from .styles import STYLESHEET
from .worker import AnonymizeWorker


def _cv2_to_qpixmap(image: np.ndarray, max_w: int = 800, max_h: int = 600) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap, scaled to fit."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


class ImagePreview(QLabel):
    """Drag-and-drop image preview area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imagePreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAcceptDrops(True)
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText("Drop an image here\nor use the buttons below")
        self.setStyleSheet(self.styleSheet())

    def show_image(self, pixmap: QPixmap):
        self.setPixmap(pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def clear_image(self):
        self.clear()
        self._set_placeholder()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            parent = self.parent()
            while parent and not isinstance(parent, MainWindow):
                parent = parent.parent()
            if parent:
                parent.load_single_image(path)


class MainWindow(QMainWindow):
    """PHI Face Anonymizer main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PHI Face Anonymizer")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self._source_image: Optional[np.ndarray] = None
        self._source_path: Optional[str] = None
        self._result: Optional[AnonymizationResult] = None
        self._folder_paths: list[Path] = []
        self._worker: Optional[AnonymizeWorker] = None

        self._build_ui()
        self.statusBar().showMessage("Ready — load an image or folder to begin")

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        main_layout.addWidget(self._build_sidebar())

        # Content area
        main_layout.addWidget(self._build_content(), stretch=1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # Title
        title = QLabel("PHI Face\nAnonymizer")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        # ---- Input section ----
        layout.addWidget(self._section_label("Input"))

        btn_load = QPushButton("Load Image")
        btn_load.setToolTip("Open a single image file")
        btn_load.clicked.connect(self._on_load_image)
        layout.addWidget(btn_load)

        btn_folder = QPushButton("Load Folder")
        btn_folder.setToolTip("Select a folder of images to batch-process")
        btn_folder.clicked.connect(self._on_load_folder)
        layout.addWidget(btn_folder)

        self._lbl_input_info = QLabel("")
        self._lbl_input_info.setWordWrap(True)
        layout.addWidget(self._lbl_input_info)

        # ---- Method section ----
        layout.addWidget(self._section_label("Anonymization"))

        layout.addWidget(QLabel("Method"))
        self._combo_method = QComboBox()
        self._combo_method.addItems(["blur", "pixelate", "blackbox"])
        self._combo_method.setToolTip("How to obscure detected faces")
        layout.addWidget(self._combo_method)

        layout.addWidget(QLabel("Blur Strength"))
        self._slider_blur = QSlider(Qt.Orientation.Horizontal)
        self._slider_blur.setRange(11, 199)
        self._slider_blur.setValue(99)
        self._slider_blur.setSingleStep(2)
        self._slider_blur.setToolTip("Gaussian kernel size (higher = more blur)")
        self._lbl_blur_val = QLabel("99")
        self._slider_blur.valueChanged.connect(self._on_blur_changed)
        row = QHBoxLayout()
        row.addWidget(self._slider_blur)
        row.addWidget(self._lbl_blur_val)
        layout.addLayout(row)

        # ---- Detection section ----
        layout.addWidget(self._section_label("Detection"))

        layout.addWidget(QLabel("Detector"))
        self._combo_detector = QComboBox()
        self._combo_detector.addItems(["mediapipe", "opencv_dnn", "auto"])
        self._combo_detector.setToolTip("Face detection backend")
        layout.addWidget(self._combo_detector)

        layout.addWidget(QLabel("Confidence"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.1, 1.0)
        self._spin_conf.setValue(0.5)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setToolTip("Minimum detection confidence (lower catches more faces)")
        layout.addWidget(self._spin_conf)

        layout.addWidget(QLabel("Padding"))
        self._spin_padding = QDoubleSpinBox()
        self._spin_padding.setRange(1.0, 2.5)
        self._spin_padding.setValue(1.3)
        self._spin_padding.setSingleStep(0.1)
        self._spin_padding.setToolTip("Bounding box expansion (1.3 = 30% larger)")
        layout.addWidget(self._spin_padding)

        self._chk_fallback = QCheckBox("Detector fallback cascade")
        self._chk_fallback.setChecked(True)
        self._chk_fallback.setToolTip("Try next detector if current finds 0 faces")
        layout.addWidget(self._chk_fallback)

        self._chk_exif = QCheckBox("Strip EXIF metadata")
        self._chk_exif.setChecked(True)
        self._chk_exif.setToolTip("Remove EXIF data from output (recommended for PHI)")
        layout.addWidget(self._chk_exif)

        layout.addStretch()

        # ---- Action buttons ----
        self._btn_run = QPushButton("Anonymize")
        self._btn_run.setObjectName("primaryButton")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        layout.addWidget(self._btn_run)

        self._btn_save = QPushButton("Save Result")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        layout.addWidget(self._btn_save)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        return sidebar

    def _build_content(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Stats bar
        stats_row = QHBoxLayout()
        self._stat_faces = self._stat_card("0", "Faces Detected")
        self._stat_status = self._stat_card("—", "Status")
        self._stat_size = self._stat_card("—", "Image Size")
        stats_row.addWidget(self._stat_faces)
        stats_row.addWidget(self._stat_status)
        stats_row.addWidget(self._stat_size)
        layout.addLayout(stats_row)

        # Before / After preview
        preview_row = QHBoxLayout()

        self._preview_before = ImagePreview()
        self._preview_after = ImagePreview()
        self._preview_after.setText("Anonymized result\nwill appear here")

        before_box = QVBoxLayout()
        before_lbl = QLabel("Original")
        before_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        before_lbl.setStyleSheet("font-weight: bold; color: #f38ba8;")
        before_box.addWidget(before_lbl)
        before_box.addWidget(self._preview_before, stretch=1)

        after_box = QVBoxLayout()
        after_lbl = QLabel("Anonymized")
        after_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        after_lbl.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        after_box.addWidget(after_lbl)
        after_box.addWidget(self._preview_after, stretch=1)

        preview_row.addLayout(before_box, stretch=1)
        preview_row.addLayout(after_box, stretch=1)
        layout.addLayout(preview_row, stretch=1)

        return content

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setObjectName("sectionLabel")
        return lbl

    def _stat_card(self, value: str, label: str) -> QWidget:
        card = QWidget()
        card.setObjectName("statsCard")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(12, 8, 12, 8)

        val = QLabel(value)
        val.setObjectName("statsValue")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl = QLabel(label)
        lbl.setObjectName("statsLabel")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        vbox.addWidget(val)
        vbox.addWidget(lbl)

        # Store ref for updates
        card._val_label = val  # type: ignore[attr-defined]
        return card

    def _update_stat(self, card: QWidget, value: str):
        card._val_label.setText(value)  # type: ignore[attr-defined]

    # ---- Slots -------------------------------------------------------------

    def _on_blur_changed(self, val: int):
        # Ensure odd
        if val % 2 == 0:
            val += 1
            self._slider_blur.setValue(val)
        self._lbl_blur_val.setText(str(val))

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.dcm);;All Files (*)"
        )
        if path:
            self.load_single_image(path)

    def load_single_image(self, path: str):
        """Load and display a single image."""
        try:
            from ..io_handler import load_image
            img = load_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Cannot load image:\n{exc}")
            return

        self._source_image = img
        self._source_path = path
        self._folder_paths = []
        self._result = None

        h, w = img.shape[:2]
        self._preview_before.show_image(_cv2_to_qpixmap(img, 560, 500))
        self._preview_after.clear_image()
        self._lbl_input_info.setText(f"Loaded: {Path(path).name}")
        self._update_stat(self._stat_size, f"{w}x{h}")
        self._update_stat(self._stat_faces, "—")
        self._update_stat(self._stat_status, "Loaded")
        self._btn_run.setEnabled(True)
        self._btn_save.setEnabled(False)
        self.statusBar().showMessage(f"Loaded {Path(path).name} ({w}x{h})")

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        from ..utils import collect_image_paths
        paths = collect_image_paths(Path(folder), recursive=True)
        if not paths:
            QMessageBox.information(self, "No Images", "No supported images found in folder.")
            return

        self._folder_paths = paths
        self._source_image = None
        self._source_path = None
        self._result = None

        # Show first image as preview
        try:
            from ..io_handler import load_image
            first = load_image(paths[0])
            self._preview_before.show_image(_cv2_to_qpixmap(first, 560, 500))
        except Exception:
            self._preview_before.clear_image()

        self._preview_after.clear_image()
        self._lbl_input_info.setText(f"Folder: {Path(folder).name}\n{len(paths)} images")
        self._update_stat(self._stat_size, f"{len(paths)} files")
        self._update_stat(self._stat_faces, "—")
        self._update_stat(self._stat_status, "Ready")
        self._btn_run.setEnabled(True)
        self._btn_save.setEnabled(False)
        self.statusBar().showMessage(f"Loaded folder with {len(paths)} images")

    def _build_anonymizer(self) -> FaceAnonymizer:
        blur_val = self._slider_blur.value()
        if blur_val % 2 == 0:
            blur_val += 1
        return FaceAnonymizer(
            method=self._combo_method.currentText(),
            blur_strength=blur_val,
            padding=self._spin_padding.value(),
            detector=self._combo_detector.currentText(),
            confidence_threshold=self._spin_conf.value(),
            fallback=self._chk_fallback.isChecked(),
            strip_exif=self._chk_exif.isChecked(),
        )

    def _on_run(self):
        if self._folder_paths:
            self._run_folder()
        elif self._source_image is not None:
            self._run_single()

    def _run_single(self):
        self._btn_run.setEnabled(False)
        self.statusBar().showMessage("Anonymizing...")
        self._update_stat(self._stat_status, "Processing...")

        anon = self._build_anonymizer()
        result = anon.process(self._source_image)

        self._result = result
        if result.success and result.image is not None:
            self._preview_after.show_image(_cv2_to_qpixmap(result.image, 560, 500))
            self._update_stat(self._stat_faces, str(result.faces_detected))
            self._update_stat(self._stat_status, "Done")
            self._btn_save.setEnabled(True)
            self.statusBar().showMessage(
                f"Anonymized — {result.faces_detected} face(s) detected"
            )
        else:
            self._update_stat(self._stat_status, "Error")
            QMessageBox.warning(self, "Error", f"Failed:\n{result.error}")

        self._btn_run.setEnabled(True)

    def _run_folder(self):
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not out_dir:
            return

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(self._folder_paths))
        self._progress.setValue(0)
        self._update_stat(self._stat_status, "Processing...")
        self.statusBar().showMessage("Processing folder...")

        anon = self._build_anonymizer()
        self._worker = AnonymizeWorker(anon, self._folder_paths, Path(out_dir))
        self._worker.progress.connect(self._on_folder_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.finished_all.connect(self._on_folder_done)
        self._worker.start()

    def _on_folder_progress(self, current: int, total: int):
        self._progress.setValue(current)
        self.statusBar().showMessage(f"Processing {current}/{total}...")

    def _on_image_done(self, result: AnonymizationResult):
        if result.success and result.image is not None:
            self._preview_after.show_image(_cv2_to_qpixmap(result.image, 560, 500))
            self._update_stat(self._stat_faces, str(result.faces_detected))

    def _on_folder_done(self, results: list):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._worker = None

        total = len(results)
        ok = sum(1 for r in results if r.success)
        faces = sum(r.faces_detected for r in results)
        self._update_stat(self._stat_status, "Done")
        self._update_stat(self._stat_faces, str(faces))
        self.statusBar().showMessage(
            f"Batch complete — {ok}/{total} images processed, {faces} faces anonymized"
        )
        QMessageBox.information(
            self, "Batch Complete",
            f"Processed {ok}/{total} images.\n{faces} total faces anonymized."
        )

    def _on_save(self):
        if self._result is None or self._result.image is None:
            return

        default_name = ""
        if self._source_path:
            p = Path(self._source_path)
            default_name = f"{p.stem}_anonymized{p.suffix}"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Anonymized Image", default_name,
            "JPEG (*.jpg);;PNG (*.png);;All Files (*)"
        )
        if path:
            from ..io_handler import save_image
            save_image(self._result.image, path, strip_exif=self._chk_exif.isChecked())
            self.statusBar().showMessage(f"Saved to {path}")


def main():
    """Launch the GUI application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
