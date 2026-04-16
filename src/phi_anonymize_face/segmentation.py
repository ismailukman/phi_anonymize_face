"""Face segmentation using MediaPipe Face Mesh for precise face masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .result import BoundingBox

# MediaPipe Face Mesh face oval indices — tight contour tracing
# the jawline, cheeks, and forehead hairline.
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Inner face region — tighter contour excluding hairline/ears.
# Traces eyebrows → temples → jawline only.
_INNER_FACE_INDICES = [
    70, 63, 105, 66, 107,  # left eyebrow top
    336, 296, 334, 293, 300,  # right eyebrow top
    # right cheek down to jaw
    383, 372, 345, 352, 376, 433, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150,
    # left cheek up from jaw
    136, 172, 213, 147, 123, 116, 143, 156, 70,
]


def create_face_mask(
    image: np.ndarray,
    box: BoundingBox,
    padding: float = 1.0,
) -> np.ndarray:
    """Create a precise face-shaped binary mask using MediaPipe Face Mesh.

    Falls back to an elliptical mask if Face Mesh fails on the region.

    Args:
        image: BGR image.
        box: Detected face bounding box.
        padding: Scale factor around contour (1.0 = exact landmarks).

    Returns:
        Binary mask (uint8, 0 or 255) same size as image.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    landmarks = _get_face_landmarks(image, box)
    if landmarks is not None:
        _draw_face_contour(mask, landmarks, padding)
    else:
        _draw_ellipse_mask(mask, box, padding)

    return mask


def _get_face_landmarks(
    image: np.ndarray, box: BoundingBox
) -> list[tuple[int, int]] | None:
    """Run Face Mesh on a cropped face region."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
    except ImportError:
        return None

    h, w = image.shape[:2]
    margin = 0.2
    x1 = max(0, int(box.x - box.w * margin))
    y1 = max(0, int(box.y - box.h * margin))
    x2 = min(w, int(box.x2 + box.w * margin))
    y2 = min(h, int(box.y2 + box.h * margin))
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    model_path = _find_face_mesh_model()
    if model_path is None:
        return None

    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        landmarker = FaceLandmarker.create_from_options(options)

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_crop
        )
        result = landmarker.detect(mp_image)
        landmarker.close()

        if not result.face_landmarks:
            return None

        ch, cw = crop.shape[:2]
        landmarks = []
        for lm in result.face_landmarks[0]:
            px = int(lm.x * cw) + x1
            py = int(lm.y * ch) + y1
            landmarks.append((px, py))
        return landmarks

    except Exception:
        return None


def _find_face_mesh_model() -> str | None:
    """Locate or download the Face Landmarker model."""
    import urllib.request

    cache_dir = Path.home() / ".cache" / "phi_anonymize_face"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        cache_dir / "face_landmarker_v2_with_blendshapes.task"
    )

    if model_path.exists():
        return str(model_path)

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/"
        "face_landmarker.task"
    )
    try:
        urllib.request.urlretrieve(url, str(model_path))
        return str(model_path)
    except Exception:
        return None


def _draw_face_contour(
    mask: np.ndarray,
    landmarks: list[tuple[int, int]],
    padding: float,
) -> None:
    """Draw the inner face contour onto the mask."""
    # Try inner face indices first (tighter)
    pts = _collect_points(landmarks, _INNER_FACE_INDICES)
    if pts is None:
        # Fall back to full oval
        pts = _collect_points(landmarks, _FACE_OVAL_INDICES)
    if pts is None:
        return

    # Only scale if padding > 1.0, and use modest scaling
    if padding > 1.0:
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        pts_f = pts.astype(np.float64)
        pts_f[:, 0] = cx + (pts_f[:, 0] - cx) * padding
        pts_f[:, 1] = cy + (pts_f[:, 1] - cy) * padding
        pts = pts_f.astype(np.int32)

    # Use fillPoly (not fillConvexPoly) for non-convex shapes
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)


def _collect_points(
    landmarks: list[tuple[int, int]],
    indices: list[int],
) -> np.ndarray | None:
    """Collect landmark points for the given indices."""
    pts = []
    for idx in indices:
        if idx < len(landmarks):
            pts.append(landmarks[idx])
    if len(pts) < 8:
        return None
    return np.array(pts, dtype=np.int32)


def _draw_ellipse_mask(
    mask: np.ndarray,
    box: BoundingBox,
    padding: float,
) -> None:
    """Draw an elliptical mask as fallback."""
    cx = box.x + box.w // 2
    cy = box.y + box.h // 2
    # Use 0.85 factor to keep ellipse inside the face, not the box
    ax = int(box.w * 0.85 * padding / 2)
    ay = int(box.h * 0.85 * padding / 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
