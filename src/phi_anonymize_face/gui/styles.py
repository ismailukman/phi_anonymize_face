"""Qt stylesheet for the application."""

STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
    color: #cdd6f4;
}

/* ---- Sidebar ---- */
#sidebar {
    background-color: #181825;
    border-right: 1px solid #313244;
}

#sidebar QLabel {
    color: #cdd6f4;
    font-size: 13px;
}

#appTitle {
    font-size: 20px;
    font-weight: bold;
    color: #89b4fa;
    padding: 8px 0;
}

#sectionLabel {
    font-size: 11px;
    font-weight: bold;
    color: #6c7086;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-top: 12px;
}

/* ---- Combo boxes ---- */
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}

/* ---- Sliders ---- */
QSlider::groove:horizontal {
    background: #313244;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #b4d0fb;
}

QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}

/* ---- Spin boxes ---- */
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    color: #cdd6f4;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #89b4fa;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px 16px;
    color: #cdd6f4;
    font-weight: 500;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}

QPushButton:pressed {
    background-color: #585b70;
}

#primaryButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
    font-size: 14px;
    border: none;
    padding: 10px 20px;
}

#primaryButton:hover {
    background-color: #b4d0fb;
}

#primaryButton:pressed {
    background-color: #74c7ec;
}

#primaryButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

#dangerButton {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

#dangerButton:hover {
    background-color: #f5a8be;
}

/* ---- Check boxes ---- */
QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

/* ---- Progress bar ---- */
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 4px;
}

/* ---- Status bar ---- */
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    font-size: 12px;
}

/* ---- Scroll area ---- */
QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #1e1e2e;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ---- Image preview area ---- */
#imagePreview {
    background-color: #11111b;
    border: 2px dashed #313244;
    border-radius: 12px;
}

#imagePreview:hover {
    border-color: #45475a;
}

/* ---- Stats cards ---- */
#statsCard {
    background-color: #313244;
    border-radius: 8px;
    padding: 12px;
}

#statsValue {
    font-size: 24px;
    font-weight: bold;
    color: #89b4fa;
}

#statsLabel {
    font-size: 11px;
    color: #6c7086;
}

/* ---- Tooltips ---- */
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}
"""
