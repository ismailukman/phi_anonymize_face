# phi_anonymize_face

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/ismailukman/phi_anonymize_face/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ismailukman/phi_anonymize_face/actions)

**HIPAA-compliant face de-identification for medical images.**

Automatically detect and anonymize faces in clinical photographs, DICOM images, and standard image formats. Built for healthcare, research, and any workflow that requires Protected Health Information (PHI) removal from facial imagery.

Use it three ways: **Python API**, **CLI**, or **GUI desktop app**.

---

## Features

- **Multi-detector cascade** — MediaPipe (fast) → OpenCV DNN/Haar (robust) → RetinaFace (high-recall). Falls back automatically if a detector finds no faces.
- **3 anonymization methods** — Gaussian blur, pixelation, or black-box redaction.
- **Medical-grade padding** — Configurable bounding-box expansion (default 1.3x) ensures ears, hairline, and chin are fully covered.
- **Desktop GUI** — PyQt6 app with side-by-side preview, tunable parameters, drag-and-drop, and batch processing.
- **CLI tool** — `phi-anonymize` command for scripting and pipelines.
- **DICOM support** — Read/write DICOM pixel data (optional `pydicom` extra).
- **EXIF stripping** — Removes metadata by default (EXIF can contain PHI).
- **Batch processing** — Parallel folder processing with progress bar (CLI, GUI, and API).
- **Audit logging** — CSV trail of every processed image for compliance.
- **Verification mode** — Re-run detection on output to confirm no residual faces.
- **Accepts images or numpy arrays** — Integrate into any Python pipeline.

---

## Installation

**Core library (API + CLI):**
```bash
pip install phi-anonymize-face
```

**With GUI:**
```bash
pip install phi-anonymize-face[gui]
```

**With DICOM support:**
```bash
pip install phi-anonymize-face[dicom]
```

**With RetinaFace (highest recall):**
```bash
pip install phi-anonymize-face[retinaface]
```

**Everything:**
```bash
pip install phi-anonymize-face[all]
```

---

## GUI Application

Launch the desktop app:

```bash
phi-anonymize-gui
```

Or from Python:

```python
from phi_anonymize_face.gui.app import main
main()
```

### GUI Features

| Feature | Description |
|---------|-------------|
| **Side-by-side preview** | Original and anonymized images shown together |
| **Drag-and-drop** | Drop an image directly onto the preview area |
| **Tunable controls** | Method, blur strength slider, detector, confidence, padding |
| **Single image mode** | Load Image → adjust settings → Anonymize → Save |
| **Batch folder mode** | Load Folder → select output folder → processes all with progress bar |
| **Stats cards** | Live display of faces detected, processing status, image dimensions |
| **EXIF toggle** | Strip or preserve EXIF metadata |
| **Fallback toggle** | Enable/disable detector cascade |
| **Dark theme** | Modern dark UI (Catppuccin Mocha) |

### GUI Workflow

**Single image:**
1. Click **Load Image** (or drag-and-drop onto the preview area)
2. Adjust settings in the sidebar (method, blur strength, detector, etc.)
3. Click **Anonymize**
4. Review the side-by-side before/after preview
5. Click **Save Result** to export

**Batch folder:**
1. Click **Load Folder** — all supported images are discovered (including subdirectories)
2. Adjust settings in the sidebar
3. Click **Anonymize** → select an output folder
4. Watch progress bar and live preview as each image is processed
5. Summary dialog shows total faces anonymized

---

## CLI Tool

The `phi-anonymize` command is available after installation.

### Single Image

```bash
# Default blur
phi-anonymize -i patient.jpg -o anonymized.jpg

# Choose method
phi-anonymize -i photo.jpg -o out.jpg --method pixelate
phi-anonymize -i photo.jpg -o out.jpg --method blackbox

# Adjust blur strength and padding
phi-anonymize -i photo.jpg -o out.jpg --blur-strength 151 --padding 1.5
```

### Folder Batch

```bash
# Process all images in a folder
phi-anonymize -i ./patient_photos/ -o ./anonymized/

# Recursive (includes subdirectories)
phi-anonymize -i ./patient_photos/ -o ./anonymized/ --recursive
```

### Detection Options

```bash
# Use a specific detector
phi-anonymize -i photo.jpg -o out.jpg --detector opencv_dnn

# Auto cascade (try all detectors)
phi-anonymize -i photo.jpg -o out.jpg --detector auto

# Lower confidence to catch more faces
phi-anonymize -i photo.jpg -o out.jpg --confidence 0.3

# Disable fallback (use only selected detector)
phi-anonymize -i photo.jpg -o out.jpg --no-fallback
```

### Compliance Options

```bash
# Enable audit trail (CSV log of every processed image)
phi-anonymize -i ./photos/ -o ./safe/ --audit-log audit.csv

# Verify output (re-run detection on anonymized images — should find 0 faces)
phi-anonymize -i photo.jpg -o safe.jpg --verify

# Keep EXIF metadata (not recommended for PHI)
phi-anonymize -i photo.jpg -o out.jpg --keep-exif
```

### Full CLI Reference

```
Usage: phi-anonymize [OPTIONS]

Options:
  -i, --input TEXT        Image or folder path (required)
  -o, --output TEXT       Output path (required)
  -m, --method TEXT       blur | pixelate | blackbox (default: blur)
  --blur-strength INT     Blur kernel size (default: 99)
  --padding FLOAT         Bounding-box padding factor (default: 1.3)
  --detector TEXT         mediapipe | opencv_dnn | retinaface | auto (default: mediapipe)
  --confidence FLOAT      Min detection confidence (default: 0.5)
  --no-fallback           Disable detector cascade fallback
  --recursive             Recurse into subdirectories (folder mode)
  --keep-exif             Preserve EXIF metadata
  --audit-log TEXT        Path for CSV audit log
  --verify                Verify output images have no residual faces
  --version               Show version
  -h, --help              Show help
```

---

## Python API

### Quick Start

```python
from phi_anonymize_face import anonymize_image, anonymize_folder, FaceAnonymizer

# 1. Quick single-image anonymization
result = anonymize_image("patient_photo.jpg", method="blur", output_path="anon.jpg")
print(f"Faces found: {result.faces_detected}")

# 2. Quick folder batch
results = anonymize_folder("input_dir/", output_dir="output_dir/", method="pixelate")

# 3. Configured anonymizer
anon = FaceAnonymizer(
    method="blur",
    blur_strength=99,
    padding=1.3,
    detector="mediapipe",      # or "opencv_dnn", "retinaface", "auto"
    confidence_threshold=0.5,
    fallback=True,             # cascade to next detector if 0 faces found
    audit_log="audit.csv",
)
result = anon.process("patient.jpg", output_path="anonymized.jpg")
results = anon.process_folder("photos/", output_dir="safe_photos/")
```

### Working with Results

```python
result = anonymize_image("photo.jpg", output_path="anon.jpg")

if result.success:
    print(f"Faces: {result.faces_detected}")
    for box in result.bounding_boxes:
        print(f"  x={box.x}, y={box.y}, w={box.w}, h={box.h}, conf={box.confidence:.2f}")
    # result.image is the anonymized numpy array (BGR)
else:
    print(f"Error: {result.error}")
```

### Using with Numpy Arrays

```python
import cv2
from phi_anonymize_face import anonymize_image

img = cv2.imread("photo.jpg")
result = anonymize_image(img, method="pixelate")
cv2.imwrite("anon.jpg", result.image)
```

---

## How Detection Cascade Works

```
Input Image
    │
    ▼
┌─────────────┐   faces found?   ┌──────────┐
│  MediaPipe   │ ──── YES ──────► │  Apply    │
│  (primary)   │                  │  method   │
└─────┬───────┘                  └──────────┘
      │ NO
      ▼
┌─────────────┐   faces found?   ┌──────────┐
│  OpenCV DNN  │ ──── YES ──────► │  Apply    │
│  (fallback)  │                  │  method   │
└─────┬───────┘                  └──────────┘
      │ NO
      ▼
┌─────────────┐   faces found?   ┌──────────┐
│  RetinaFace  │ ──── YES ──────► │  Apply    │
│  (optional)  │                  │  method   │
└─────┬───────┘                  └──────────┘
      │ NO
      ▼
  Return image unchanged (log warning)
```

Set `fallback=False` to use only the selected detector.

---

## Anonymization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `blur` | Gaussian blur (configurable kernel) | Default — preserves image context |
| `pixelate` | Block-based downscale/upscale | Visible anonymization indicator |
| `blackbox` | Solid black fill | Maximum privacy, no facial info retained |

---

## DICOM Support

```python
from phi_anonymize_face import anonymize_image

# Works with .dcm files when pydicom is installed
result = anonymize_image("scan.dcm", output_path="anon.dcm")
```

Install with `pip install phi-anonymize-face[dicom]`.

---

## API Reference

### `FaceAnonymizer`

The main class for configuring and running face anonymization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"blur"` | Anonymization method: `"blur"`, `"pixelate"`, `"blackbox"` |
| `blur_strength` | `int` | `99` | Kernel size for blur / block size for pixelate |
| `padding` | `float` | `1.3` | Bounding-box expansion factor (1.3 = 30% larger) |
| `detector` | `str` | `"mediapipe"` | Backend: `"mediapipe"`, `"opencv_dnn"`, `"retinaface"`, `"auto"` |
| `confidence_threshold` | `float` | `0.5` | Minimum detection confidence to keep a face |
| `fallback` | `bool` | `True` | Cascade to next detector if current finds 0 faces |
| `strip_exif` | `bool` | `True` | Remove EXIF metadata from output (recommended for PHI) |
| `audit_log` | `str\|None` | `None` | Path for CSV audit trail file |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `process()` | `process(source, output_path=None) → AnonymizationResult` | Anonymize a single image (path, Path, or numpy array) |
| `process_folder()` | `process_folder(folder, output_dir, recursive=False) → list[AnonymizationResult]` | Batch-process all images in a folder |
| `verify()` | `verify(source) → bool` | Returns `True` if no faces are detected (post-anonymization check) |

---

### `anonymize_image`

```python
anonymize_image(source, method="blur", output_path=None, **kwargs) → AnonymizationResult
```

Quick function for single images. All `**kwargs` are forwarded to `FaceAnonymizer`.

---

### `anonymize_folder`

```python
anonymize_folder(folder, output_dir, method="blur", recursive=False, **kwargs) → list[AnonymizationResult]
```

Quick function for batch processing. All `**kwargs` are forwarded to `FaceAnonymizer`.

---

### `AnonymizationResult`

Dataclass returned by all processing methods.

| Field | Type | Description |
|-------|------|-------------|
| `image` | `np.ndarray\|None` | The anonymized image (BGR numpy array) |
| `faces_detected` | `int` | Number of faces found |
| `bounding_boxes` | `list[BoundingBox]` | Padded bounding boxes of detected faces |
| `output_path` | `str\|None` | Path where the image was saved (if requested) |
| `success` | `bool` | Whether processing completed without error |
| `error` | `str\|None` | Error message if `success` is `False` |

---

## Step-by-Step Usage Guide

### Option 1: GUI (Desktop App) — No coding required

1. **Install:** `pip install phi-anonymize-face[gui]`
2. **Launch:** Run `phi-anonymize-gui` in your terminal
3. **Load an image:** Click **Load Image** in the sidebar, or drag-and-drop onto the preview area
4. **Adjust settings:** Use the sidebar controls to pick a method (blur/pixelate/blackbox), set blur strength, choose a detector, and adjust confidence and padding
5. **Anonymize:** Click the blue **Anonymize** button
6. **Review:** Compare the original and anonymized images side-by-side
7. **Save:** Click **Save Result** and choose where to export

**For a folder of images:**
1. Click **Load Folder** instead of Load Image
2. Adjust settings as needed
3. Click **Anonymize** → select an output folder when prompted
4. Watch the progress bar — each image previews as it completes
5. A summary dialog appears when done

---

### Option 2: CLI (Command Line) — For scripting and automation

1. **Install:** `pip install phi-anonymize-face`
2. **Single image:** `phi-anonymize -i photo.jpg -o anonymized.jpg`
3. **Choose a method:** Add `--method pixelate` or `--method blackbox`
4. **Adjust blur:** Add `--blur-strength 151` (higher = more blur)
5. **Batch a folder:** `phi-anonymize -i ./photos/ -o ./safe/ --recursive`
6. **Add audit log:** Append `--audit-log audit.csv`
7. **Verify output:** Append `--verify` to confirm no residual faces

---

### Option 3: Python API — For integration into code/pipelines

1. **Install:** `pip install phi-anonymize-face`
2. **Quick anonymize:**
   ```python
   from phi_anonymize_face import anonymize_image
   result = anonymize_image("photo.jpg", output_path="anon.jpg")
   print(f"Faces blurred: {result.faces_detected}")
   ```
3. **Choose method:**
   ```python
   anonymize_image("photo.jpg", method="pixelate", output_path="pix.jpg")
   ```
4. **Batch a folder:**
   ```python
   from phi_anonymize_face import anonymize_folder
   results = anonymize_folder("./photos/", output_dir="./safe/")
   ```
5. **Fine-tune settings:**
   ```python
   from phi_anonymize_face import FaceAnonymizer
   anon = FaceAnonymizer(
       method="blur", blur_strength=99, padding=1.3,
       detector="mediapipe", confidence_threshold=0.5,
       fallback=True, audit_log="audit.csv",
   )
   result = anon.process("photo.jpg", output_path="safe.jpg")
   ```
6. **Use with numpy arrays:**
   ```python
   import cv2
   from phi_anonymize_face import anonymize_image
   img = cv2.imread("photo.jpg")
   result = anonymize_image(img, method="blur")
   cv2.imwrite("anon.jpg", result.image)
   ```

---

## Development

```bash
git clone https://github.com/ismailukman/phi_anonymize_face.git
cd phi_anonymize_face
pip install -e ".[dev,gui]"
pytest
ruff check src/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE).
