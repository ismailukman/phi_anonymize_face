# Changelog

## 0.1.0 (2026-04-16)

- Initial release.
- Face detection via MediaPipe, OpenCV DNN (YuNet/Haar), and RetinaFace.
- Cascading detector fallback for maximum recall.
- Anonymization methods: Gaussian blur, pixelation, black-box redaction.
- Configurable bounding-box padding for full PHI coverage.
- Single image and batch folder processing with parallel execution.
- CLI tool (`phi-anonymize`).
- DICOM support (optional `pydicom`).
- EXIF stripping by default for PHI safety.
- CSV audit logging.
- Post-anonymization verification mode.
