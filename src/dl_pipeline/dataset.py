"""Dataset assembly for the DL pipeline.

Builds a pandas DataFrame index of (frame, landmarks, label, video) and
exposes a group-aware split (videos never appear in both train and test).
Per-frame and windowed torch Datasets crop a 224x224 face ROI on demand.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config

_VIDEO_RE = re.compile(r"(.+)_f\d+$")


def _video_id(stem: str) -> str:
    m = _VIDEO_RE.match(stem)
    return m.group(1) if m else stem


def build_index(
    frames_root: Path,
    landmarks_root: Path,
    labels: tuple[str, ...] = ("trigger", "non_trigger"),
) -> pd.DataFrame:
    """Pair frame .jpg files with their _px.npy landmark companions.

    Returns DataFrame with columns: frame_path, landmarks_path, label, video.
    Skips frames whose landmark companion is missing.
    """
    rows = []
    for label in labels:
        if label not in config.LABEL_MAP:
            raise ValueError(f"Unknown label: {label}")
        label_int = config.LABEL_MAP[label]
        lm_dir = Path(landmarks_root) / label
        fr_dir = Path(frames_root) / label
        if not lm_dir.exists():
            continue
        for px_path in sorted(lm_dir.glob("*_px.npy")):
            stem = px_path.stem[:-3]  # strip "_px"
            frame_path = fr_dir / f"{stem}.jpg"
            if not frame_path.exists():
                frame_path = fr_dir / f"{stem}.png"
                if not frame_path.exists():
                    continue
            rows.append({
                "frame_path": frame_path,
                "landmarks_path": px_path,
                "label": label_int,
                "video": _video_id(stem),
            })
    if not rows:
        raise RuntimeError(f"No frames indexed under {frames_root} / {landmarks_root}")
    return pd.DataFrame(rows)


def split_indices(
    df: pd.DataFrame,
    test_size: float = config.SPLIT_TEST_SIZE,
    random_state: int = config.SPLIT_RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Group-aware train/test split (no video appears in both).

    Locked to GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    to match Member 3's split exactly.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.zeros(len(df)), df["label"].values, df["video"].values))
    return train_idx, test_idx
