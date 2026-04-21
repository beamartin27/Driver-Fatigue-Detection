"""
dataset.py
==========
Member 3 — Dataset builder.

Pairs each landmark .npy file produced by Member 2 with its corresponding
frame image, extracts EAR/MAR/HOG features, and assembles the (X, y, groups)
arrays used for training and evaluation.

Labels:
    trigger     -> 1  (fatigue-indicating / positive class)
    non_trigger -> 0  (baseline / negative class)

Group IDs are source-video stems (e.g. "WIN_20260322_11_01_32_Pro") parsed
from the frame filename.  Downstream code uses these groups to ensure no
source video appears in both train and test splits.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

import cv2

from .features import extract_frame_features

# ── Label mapping ─────────────────────────────────────────────────────────────

LABEL_MAP: dict[str, int] = {
    "trigger":     1,
    "non_trigger": 0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _source_video(stem: str) -> str:
    """
    Extract source video ID from a frame file stem.

    Example: "WIN_20260322_11_01_32_Pro_f00012" -> "WIN_20260322_11_01_32_Pro"
    Falls back to the full stem if the pattern does not match.
    """
    m = re.match(r"(.+)_f\d+$", stem)
    return m.group(1) if m else stem


# ── Public API ────────────────────────────────────────────────────────────────

def build_dataset(
    frames_root: Path,
    landmarks_root: Path,
    labels: tuple[str, ...] = ("trigger", "non_trigger"),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan landmark folders, pair with frame images, extract features.

    Expected folder layout (produced by landmark_extractor.py and extract_frames.py):

        landmarks_root/
            trigger/
                WIN_..._f00000.npy        <- (478, 3) normalised coords
                WIN_..._f00000_px.npy     <- (478, 2) pixel coords
                ...
            non_trigger/
                ...

        frames_root/
            trigger/
                WIN_..._f00000.jpg
                ...
            non_trigger/
                ...

    A frame is skipped (with a counter) when:
      - the matching _px.npy file is missing,
      - the matching .jpg / .png image is missing,
      - cv2.imread returns None,
      - feature extraction returns None (degenerate face ROI).

    Parameters
    ----------
    frames_root    : Path to data/frames/
    landmarks_root : Path to data/landmarks/
    labels         : Ordered tuple of class folder names to process.
    verbose        : Print per-class progress and final summary.

    Returns
    -------
    X      : (N, F) float32  — feature matrix
    y      : (N,)   int32    — label vector  (1=trigger, 0=non_trigger)
    groups : (N,)   object   — source-video ID per sample (for GroupShuffleSplit)
    """
    X_list:      list[np.ndarray] = []
    y_list:      list[int]        = []
    groups_list: list[str]        = []

    for label in labels:
        lm_dir    = landmarks_root / label
        frame_dir = frames_root    / label

        if not lm_dir.exists():
            if verbose:
                print(f"[WARN] Landmark directory not found, skipping: {lm_dir}")
            continue

        # Collect normalised landmark files (exclude _px.npy companions)
        norm_files = sorted(
            f for f in lm_dir.glob("*.npy")
            if not f.name.endswith("_px.npy")
        )

        if not norm_files:
            if verbose:
                print(f"[WARN] No .npy files found in {lm_dir}")
            continue

        y_val   = LABEL_MAP[label]
        accepted = 0
        skipped  = 0

        if verbose:
            print(f"\n[{label.upper()}] {len(norm_files)} landmark files")

        for norm_path in norm_files:
            stem = norm_path.stem   # e.g. "WIN_20260322_11_01_32_Pro_f00000"

            # Pixel landmark companion
            px_path = lm_dir / f"{stem}_px.npy"
            if not px_path.exists():
                skipped += 1
                continue

            # Frame image
            frame_path = frame_dir / f"{stem}.jpg"
            if not frame_path.exists():
                frame_path = frame_dir / f"{stem}.png"
            if not frame_path.exists():
                skipped += 1
                continue

            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                skipped += 1
                continue

            landmarks_px = np.load(str(px_path))   # (478, 2) int32

            feat = extract_frame_features(frame_bgr, landmarks_px)
            if feat is None:
                skipped += 1
                continue

            X_list.append(feat)
            y_list.append(y_val)
            groups_list.append(_source_video(stem))
            accepted += 1

        if verbose:
            print(f"  accepted: {accepted}   skipped: {skipped}")

    if not X_list:
        raise RuntimeError(
            "No features extracted. Check that frames_root and landmarks_root "
            "point to the correct directories and contain matching files."
        )

    X      = np.stack(X_list,  axis=0).astype(np.float32)
    y      = np.array(y_list,  dtype=np.int32)
    groups = np.array(groups_list, dtype=object)

    if verbose:
        n_trigger     = int((y == LABEL_MAP["trigger"]).sum())
        n_non_trigger = int((y == LABEL_MAP["non_trigger"]).sum())
        n_groups      = len(set(groups_list))
        print(
            f"\nDataset summary"
            f"\n  Total samples : {len(X)}"
            f"\n  Feature dims  : {X.shape[1]}"
            f"\n  trigger (1)   : {n_trigger}"
            f"\n  non_trigger(0): {n_non_trigger}"
            f"\n  Unique videos : {n_groups}"
        )

    return X, y, groups
