# PyInstaller spec for the CLI.
# Build:  pyinstaller packaging/phi-anonymize.spec --noconfirm
from PyInstaller.utils.hooks import collect_all, collect_data_files

mp_datas, mp_binaries, mp_hiddenimports = collect_all("mediapipe")
cv2_datas = collect_data_files("cv2")

a = Analysis(
    ["../src/phi_anonymize_face/__main__.py"],
    pathex=["../src"],
    binaries=mp_binaries,
    datas=mp_datas + cv2_datas,
    hiddenimports=mp_hiddenimports + [
        "phi_anonymize_face",
        "phi_anonymize_face.detectors.mediapipe_detector",
        "phi_anonymize_face.detectors.opencv_dnn_detector",
        "phi_anonymize_face.methods.blur",
        "phi_anonymize_face.methods.pixelate",
        "phi_anonymize_face.methods.blackbox",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt6", "insightface", "onnxruntime", "pydicom"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="phi-anonymize",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="phi-anonymize",
)
