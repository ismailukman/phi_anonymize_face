# Changelog

## 0.2.2 (2026-05-02)

- Prebuilt standalone binaries for Windows and macOS (Apple Silicon) — both GUI (`phi-anonymize-gui`) and CLI (`phi-anonymize`) — published as GitHub Release assets. No Python install required.
- Add `release.yml` GitHub Actions workflow: tag-triggered, builds binaries on Windows + macOS arm64, builds wheel/sdist on Linux, and publishes a GitHub Release with all artifacts attached. (Intel mac binaries dropped — `macos-13` hosted runners routinely time out at the 24h queue limit. Intel mac users: `pip install phi-anonymize-face[gui]`.)
- Add PyInstaller specs (`packaging/phi-anonymize-gui.spec`, `packaging/phi-anonymize.spec`).
- Tighten CLI `--detector` validation (now `click.Choice` — invalid values are rejected up front instead of silently falling through the cascade).
- README rewrite: three-tier install ladder (download binary → pipx → pip), per-OS binary instructions, Gatekeeper/SmartScreen workarounds.

## 0.2.0 (2026-04-16)

- Add precise face-contour masking mode (`mask_mode="face"`).
- MediaPipe Face Mesh (468 landmarks) for accurate face-only blur.
- Tighter inner-face contour — no spill onto hair, ears, or background.
- Fallback to elliptical mask when Face Mesh is unavailable.
- Add `--mask-mode` CLI option and GUI dropdown.
- Add PyQt6 desktop GUI with side-by-side preview and batch processing.
- GUI screenshot added to README.
- Fix tests for uniform-image edge case.

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
