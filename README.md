# phi_anonymize_face

[![PyPI](https://img.shields.io/pypi/v/phi-anonymize-face)](https://pypi.org/project/phi-anonymize-face/)
[![Python](https://img.shields.io/pypi/pyversions/phi-anonymize-face)](https://pypi.org/project/phi-anonymize-face/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/ismailukman/phi_anonymize_face/actions/workflows/ci.yml/badge.svg)](https://github.com/ismailukman/phi_anonymize_face/actions)

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

### `FaceAnonymizer(method, blur_strength, padding, detector, confidence_threshold, fallback, strip_exif, audit_log)`

Main class. All parameters are optional with sensible defaults.

### `anonymize_image(source, method, output_path, **kwargs) → AnonymizationResult`

Quick function for single images.

### `anonymize_folder(folder, output_dir, method, recursive, **kwargs) → list[AnonymizationResult]`

Quick function for batch processing.

### `AnonymizationResult`

Dataclass with: `image` (ndarray), `faces_detected` (int), `bounding_boxes` (list), `output_path` (str|None), `success` (bool), `error` (str|None).

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
