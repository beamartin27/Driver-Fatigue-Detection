"""
features.py
===========
Member 3 — EAR, MAR, and HOG feature extraction.

Consumes outputs from Member 2 (landmark_extractor.py):
  - landmarks_px : (478, 2) int32   pixel (u, v) coordinates
  - frame_bgr    : H×W×3  uint8    BGR image from cv2

Returns a flat float32 feature vector per frame:
  [left_ear, right_ear, avg_ear, mar,  <HOG vector>]

Landmark index constants are imported directly from the root-level
landmark_extractor.py (Member 2) to avoid duplicated definitions.
"""

import numpy as np
import cv2
from skimage.feature import hog

# Landmark subset indices — mirror of the constants in landmark_extractor.py (Member 2).
# Defined locally to avoid importing mediapipe, which is not needed by the classical pipeline.
# If Member 2 ever changes these indices, update here to match.
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]   # P1..P6 clockwise around left eye
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]   # P1..P6 clockwise around right eye
MOUTH_MAR     = [61,   39,   0, 269, 291, 405, 17, 181]  # P1..P8 outer lip

# ── HOG config ────────────────────────────────────────────────────────────────

HOG_IMAGE_SIZE      = (64, 64)   # face ROI resize target before HOG
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
# With 64×64, 8-px cells, 2-cell blocks, stride 1 → (7×7) blocks × 4 cells × 9 bins = 1764 features
HOG_BLOCK_NORM      = "L2-Hys"

HOG_FACE_PAD_RATIO  = 0.10    # 10% padding around face bounding box

FEATURE_NAMES = ["left_ear", "right_ear", "avg_ear", "mar"]  # scalar prefix; HOG appended

# ── Scalar feature helpers ────────────────────────────────────────────────────


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def compute_ear(eye_pts: np.ndarray) -> float:
    """
    Eye Aspect Ratio from 6 ordered eye landmark points (pixel coords).

    Formula: EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)

    Parameters
    ----------
    eye_pts : (6, 2) float array — points in the order defined by
              LEFT_EYE_EAR / RIGHT_EYE_EAR in landmark_extractor.py.

    Returns
    -------
    ear : float — 0.0 when the horizontal distance is degenerate.
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    numerator   = _dist(p2, p6) + _dist(p3, p5)
    denominator = 2.0 * _dist(p1, p4)
    return numerator / denominator if denominator > 1e-6 else 0.0


def compute_mar(mouth_pts: np.ndarray) -> float:
    """
    Mouth Aspect Ratio from 8 ordered mouth landmark points (pixel coords).

    Formula: MAR = (|P2-P8| + |P3-P7| + |P4-P6|) / (2 * |P1-P5|)

    Parameters
    ----------
    mouth_pts : (8, 2) float array — points in the order defined by
                MOUTH_MAR in landmark_extractor.py.

    Returns
    -------
    mar : float — 0.0 when the horizontal distance is degenerate.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_pts
    numerator   = _dist(p2, p8) + _dist(p3, p7) + _dist(p4, p6)
    denominator = 2.0 * _dist(p1, p5)
    return numerator / denominator if denominator > 1e-6 else 0.0


# ── HOG helpers ───────────────────────────────────────────────────────────────


def _extract_face_roi(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
) -> np.ndarray | None:
    """
    Crop a padded, grayscale face ROI from the frame and resize to HOG_IMAGE_SIZE.

    Uses the first 468 landmarks (face mesh without iris) to build the
    tightest bounding box, then adds HOG_FACE_PAD_RATIO padding on each side.

    Returns None if the resulting ROI is degenerate (zero area).
    """
    h, w = frame_bgr.shape[:2]
    face_pts = landmarks_px[:468]           # exclude iris points (468-477)

    x_min, y_min = face_pts.min(axis=0)
    x_max, y_max = face_pts.max(axis=0)

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    pad_x  = max(1, int(bbox_w * HOG_FACE_PAD_RATIO))
    pad_y  = max(1, int(bbox_h * HOG_FACE_PAD_RATIO))

    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(w, int(x_max) + pad_x)
    y2 = min(h, int(y_max) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    roi_gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi_gray, HOG_IMAGE_SIZE)


def compute_hog_features(face_roi_gray: np.ndarray) -> np.ndarray:
    """
    Compute a HOG descriptor from a grayscale face ROI.

    Parameters
    ----------
    face_roi_gray : (H, W) uint8 grayscale image (expected HOG_IMAGE_SIZE).

    Returns
    -------
    descriptor : (1764,) float32 HOG feature vector.
    """
    descriptor = hog(
        face_roi_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_frame_features(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
) -> np.ndarray | None:
    """
    Compute the full feature vector for one frame.

    Layout:  [left_ear, right_ear, avg_ear, mar,  <1764 HOG dims>]
                  (4 scalars)                         total = 1768

    Parameters
    ----------
    frame_bgr    : H×W×3 uint8 BGR image (as returned by cv2.imread).
    landmarks_px : (478, 2) int32 pixel landmark array from Member 2.

    Returns
    -------
    feature_vector : (1768,) float32, or None if the face ROI is degenerate.
    """
    lm = landmarks_px.astype(np.float32)

    # ── EAR ──
    left_ear  = compute_ear(lm[LEFT_EYE_EAR])
    right_ear = compute_ear(lm[RIGHT_EYE_EAR])
    avg_ear   = (left_ear + right_ear) / 2.0

    # ── MAR ──
    mar = compute_mar(lm[MOUTH_MAR])

    # ── HOG ──
    roi = _extract_face_roi(frame_bgr, landmarks_px)
    if roi is None:
        return None
    hog_vec = compute_hog_features(roi)

    scalar_block = np.array([left_ear, right_ear, avg_ear, mar], dtype=np.float32)
    return np.concatenate([scalar_block, hog_vec])
