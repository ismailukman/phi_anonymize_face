# PyInstaller spec for the GUI app.
# Build:  pyinstaller packaging/phi-anonymize-gui.spec --noconfirm
from PyInstaller.utils.hooks import collect_all, collect_data_files

mp_datas, mp_binaries, mp_hiddenimports = collect_all("mediapipe")
cv2_datas = collect_data_files("cv2")

a = Analysis(
    ["../src/phi_anonymize_face/gui/__main__.py"],
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
    excludes=["tkinter", "insightface", "onnxruntime", "pydicom"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="phi-anonymize-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="phi-anonymize-gui",
)

# macOS .app bundle
import sys
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="phi-anonymize-gui.app",
        icon=None,
        bundle_identifier="org.haleyouthfoundation.phi-anonymize-face",
        info_plist={
            "NSHighResolutionCapable": "True",
            "CFBundleShortVersionString": "0.2.2",
            "NSCameraUsageDescription": "Not used. Required by linked frameworks.",
        },
    )
