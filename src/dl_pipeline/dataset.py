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


import torch
from torch.utils.data import Dataset
from torchvision import transforms


def face_crop_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
    out_size: int = config.INPUT_SIZE,
    pad_ratio: float = config.FACE_PAD_RATIO,
) -> np.ndarray:
    """Crop a padded face ROI from frame_bgr using landmarks_px[:468]."""
    h, w = frame_bgr.shape[:2]
    face_pts = landmarks_px[:468]
    x_min, y_min = face_pts.min(axis=0)
    x_max, y_max = face_pts.max(axis=0)
    bbox_w = max(1, int(x_max - x_min))
    bbox_h = max(1, int(y_max - y_min))
    pad_x = max(1, int(bbox_w * pad_ratio))
    pad_y = max(1, int(bbox_h * pad_ratio))
    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(w, int(x_max) + pad_x)
    y2 = min(h, int(y_max) + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    roi = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(roi, (out_size, out_size))


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class FrameDataset(Dataset):
    """Per-frame dataset: loads frame, crops face, returns (image, label)."""

    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = _build_transform(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        frame_bgr = cv2.imread(str(row["frame_path"]))
        if frame_bgr is None:
            raise FileNotFoundError(row["frame_path"])
        landmarks = np.load(str(row["landmarks_path"]))
        crop_bgr = face_crop_from_landmarks(frame_bgr, landmarks)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = self.transform(crop_rgb)
        return {"image": image, "label": int(row["label"])}
