# phi_anonymize_face

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/ismailukman/phi_anonymize_face/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ismailukman/phi_anonymize_face/actions)

**HIPAA-compliant face de-identification for medical images.**

Automatically detect and anonymize faces in clinical photographs, DICOM images, and standard image formats. Built for healthcare, research, and any workflow that requires Protected Health Information (PHI) removal from facial imagery.

---

## Features

- **Multi-detector cascade** — MediaPipe (fast) → OpenCV DNN/Haar (robust) → RetinaFace (high-recall). Falls back automatically if a detector finds no faces.
- **3 anonymization methods** — Gaussian blur, pixelation, or black-box redaction.
- **Medical-grade padding** — Configurable bounding-box expansion (default 1.3x) ensures ears, hairline, and chin are fully covered.
- **DICOM support** — Read/write DICOM pixel data (optional `pydicom` extra).
- **EXIF stripping** — Removes metadata by default (EXIF can contain PHI).
- **Batch processing** — Parallel folder processing with progress bar.
- **CLI tool** — `phi-anonymize` command for scripting and pipelines.
- **Audit logging** — CSV trail of every processed image for compliance.
- **Verification mode** — Re-run detection on output to confirm no residual faces.
- **Accepts images or numpy arrays** — Integrate into any Python pipeline.

---

## Installation

```bash
pip install phi-anonymize-face
```

With DICOM support:
```bash
pip install phi-anonymize-face[dicom]
```

With RetinaFace (highest recall):
```bash
pip install phi-anonymize-face[retinaface]
```

Everything:
```bash
pip install phi-anonymize-face[all]
```

---

## Quick Start

### Python API

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

### CLI

```bash
# Single image
phi-anonymize -i patient.jpg -o anon.jpg --method blur --blur-strength 99

# Folder (recursive)
phi-anonymize -i ./patient_photos/ -o ./anonymized/ --method pixelate --recursive

# With verification (re-checks output for residual faces)
phi-anonymize -i photo.jpg -o safe.jpg --verify

# With audit trail
phi-anonymize -i ./photos/ -o ./out/ --audit-log audit.csv
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

## Development

```bash
git clone https://github.com/ismailukman/phi_anonymize_face.git
cd phi_anonymize_face
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE).
