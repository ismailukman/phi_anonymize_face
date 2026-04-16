"""Face segmentation using MediaPipe Face Mesh for precise face masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .result import BoundingBox

# MediaPipe Face Mesh face oval landmark indices (ordered for convex hull)
# These trace the outer contour of the face.
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
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
        box: Detected face bounding box (unpadded or lightly padded).
        padding: Extra padding around the face contour (1.0 = tight fit).

    Returns:
        Binary mask (uint8, 0 or 255) same size as image.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Try MediaPipe Face Mesh for precise contour
    landmarks = _get_face_landmarks(image, box)
    if landmarks is not None:
        _draw_face_contour(mask, landmarks, box, padding)
    else:
        # Fallback: elliptical mask inscribed in the bounding box
        _draw_ellipse_mask(mask, box, padding)

    return mask


def _get_face_landmarks(
    image: np.ndarray, box: BoundingBox
) -> list[tuple[int, int]] | None:
    """Run Face Mesh on a cropped face region and return landmarks."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
    except ImportError:
        return None

    # Crop with some margin for Face Mesh to work well
    h, w = image.shape[:2]
    margin = 0.3
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
    model_path = cache_dir / "face_landmarker_v2_with_blendshapes.task"

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
    box: BoundingBox,
    padding: float,
) -> None:
    """Draw the face oval contour onto the mask."""
    # Extract face oval points
    oval_pts = []
    for idx in _FACE_OVAL_INDICES:
        if idx < len(landmarks):
            oval_pts.append(landmarks[idx])

    if len(oval_pts) < 10:
        _draw_ellipse_mask(mask, box, padding)
        return

    pts = np.array(oval_pts, dtype=np.int32)

    # Apply padding by scaling outward from centroid
    if padding > 1.0:
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        pts_f = pts.astype(np.float64)
        pts_f[:, 0] = cx + (pts_f[:, 0] - cx) * padding
        pts_f[:, 1] = cy + (pts_f[:, 1] - cy) * padding
        pts = pts_f.astype(np.int32)

    cv2.fillConvexPoly(mask, pts, 255)


def _draw_ellipse_mask(
    mask: np.ndarray,
    box: BoundingBox,
    padding: float,
) -> None:
    """Draw an elliptical mask as fallback."""
    cx = box.x + box.w // 2
    cy = box.y + box.h // 2
    ax = int(box.w * padding / 2)
    ay = int(box.h * padding / 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
