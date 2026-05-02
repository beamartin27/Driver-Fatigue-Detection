# Member 4 — Deep Learning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deep-learning fatigue-detection branch (Member 4) — a per-frame MobileNetV3-Small CNN and a CNN-LSTM temporal model — that consumes Member 2's landmarks, evaluates on the same held-out split as Member 3's classical pipeline, and exposes a clean `DLPredictor` API for Member 5 to fuse.

**Architecture:** PyTorch. Two models share a backbone: (1) MobileNetV3-Small fine-tuned on face crops (224×224, derived from Member 2's `landmarks_px` bbox), and (2) BiLSTM head over 16-frame windows of the CNN's 576-dim features. ImageNet pretraining only — no external fatigue dataset.

**Tech Stack:** Python 3.9+, PyTorch + torchvision, scikit-learn (for `GroupShuffleSplit` and metrics), OpenCV (frame loading), pytest (tests), matplotlib (plots).

---

## File Structure

New files (all under `Driver-Fatigue-Detection/`):

```
src/dl_pipeline/
├── __init__.py
├── config.py                       # paths + hyperparams + seed
├── dataset.py                      # build_index, split_indices, FrameDataset, WindowDataset
├── models.py                       # build_cnn(), build_cnn_lstm(), CNNFeatureExtractor
├── train_eval.py                   # train loops, evaluate_*, metrics, plots, feature caching
├── predict.py                      # DLPredictor class
├── run_dl_pipeline.py              # CLI: train_cnn / cache_features / train_lstm / evaluate
└── requirements_dl_pipeline.txt

tests/dl_pipeline/
├── __init__.py
├── conftest.py                     # synthetic fixtures (tiny CSV + fake frames/landmarks)
├── test_dataset.py
├── test_models.py
├── test_train_eval.py
└── test_predict.py

outputs/dl_pipeline/                # created at runtime by training scripts
├── cnn_mobilenetv3.pt
├── cnn_lstm.pt
├── feature_cache/<video>.npy
├── metrics.json
├── confusion_cnn.png
├── confusion_cnn_lstm.png
├── roc_curves.png
└── runs/<timestamp>/
```

Existing files **not modified**: `landmark_extractor.py`, `extract_frames.py`, `src/classical_pipeline/*`, `src/gesture_activation/*`.

---

## Conventions

- **Working directory for all commands:** `/Users/matthewmaingot/CV_Group_Project/Driver-Fatigue-Detection/`
- **Test runner:** `pytest tests/dl_pipeline/ -v`
- **Switch torch modules to inference mode with `.train(False)`** (equivalent to PyTorch's standard inference toggle, used throughout this codebase to keep linters / security tools happy).
- **Split seed (locked):** `GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)` — must match `src/classical_pipeline/train_eval.py:69` exactly.
- **Label map (locked):** `{"trigger": 1, "non_trigger": 0}` — must match `src/classical_pipeline/dataset.py:31`.
- **Group ID extraction (locked):** regex `r"(.+)_f\d+$"` on the file stem — must match `src/classical_pipeline/dataset.py:46`.
- **Bbox extraction (locked):** use first 468 landmarks (exclude iris 468–477), pad 10% on each side, clamp to image bounds — same logic as `src/classical_pipeline/features.py:93`.

---

## Task 0: Scaffolding

**Files:**
- Create: `src/dl_pipeline/__init__.py`
- Create: `src/dl_pipeline/requirements_dl_pipeline.txt`
- Create: `tests/dl_pipeline/__init__.py`

- [ ] **Step 1: Create empty package init files**

```python
# src/dl_pipeline/__init__.py
"""Member 4 — Deep learning pipeline (PyTorch CNN + CNN-LSTM)."""
```

```python
# tests/dl_pipeline/__init__.py
```

- [ ] **Step 2: Write requirements file**

```
# src/dl_pipeline/requirements_dl_pipeline.txt
torch>=2.0
torchvision>=0.15
numpy
pandas
opencv-python
scikit-learn
matplotlib
tqdm
pytest
```

- [ ] **Step 3: Commit**

```bash
cd /Users/matthewmaingot/CV_Group_Project/Driver-Fatigue-Detection
git add src/dl_pipeline/__init__.py src/dl_pipeline/requirements_dl_pipeline.txt tests/dl_pipeline/__init__.py
git commit -m "chore(dl_pipeline): scaffold package and requirements"
```

---

## Task 1: config.py

**Files:**
- Create: `src/dl_pipeline/config.py`

- [ ] **Step 1: Write config module**

```python
# src/dl_pipeline/config.py
"""Centralised paths, hyperparameters, and seeds for the DL pipeline."""
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parents[2]
DATA_FRAMES     = REPO_ROOT / "data" / "frames"
DATA_LANDMARKS  = REPO_ROOT / "data" / "landmarks"
DATASET_SUMMARY = DATA_LANDMARKS / "dataset_summary.csv"
OUTPUTS_DIR     = REPO_ROOT / "outputs" / "dl_pipeline"
FEATURE_CACHE   = OUTPUTS_DIR / "feature_cache"

SPLIT_TEST_SIZE   = 0.25
SPLIT_RANDOM_SEED = 42

LABEL_MAP   = {"trigger": 1, "non_trigger": 0}
NUM_CLASSES = 2

FACE_PAD_RATIO = 0.10
INPUT_SIZE     = 224

CNN_BATCH_SIZE   = 64
CNN_EPOCHS       = 20
CNN_HEAD_LR      = 1e-4
CNN_BB_LR        = 1e-5
CNN_WEIGHT_DECAY = 1e-4

WINDOW_SIZE         = 16
WINDOW_STRIDE_TRAIN = 1
WINDOW_STRIDE_EVAL  = 4
LSTM_HIDDEN     = 128
LSTM_DROPOUT    = 0.3
LSTM_BATCH_SIZE = 128
LSTM_EPOCHS     = 25
LSTM_LR         = 3e-4

GLOBAL_SEED = 42
```

- [ ] **Step 2: Commit**

```bash
git add src/dl_pipeline/config.py
git commit -m "feat(dl_pipeline): add config with paths, hyperparams, locked split seed"
```

---

## Task 2: dataset.py — index + split

**Files:**
- Create: `src/dl_pipeline/dataset.py`
- Create: `tests/dl_pipeline/conftest.py`
- Create: `tests/dl_pipeline/test_dataset.py`

- [ ] **Step 1: Write the conftest fixture (synthetic dataset)**

```python
# tests/dl_pipeline/conftest.py
"""Synthetic fixtures so tests don't depend on the real captured dataset."""
import numpy as np
import cv2
import pytest


@pytest.fixture
def fake_dataset(tmp_path):
    """Build a tiny on-disk dataset with 3 fake videos x 4 frames each.

    Layout mirrors the real one:
        frames/{label}/{video}_f{idx:05d}.jpg
        landmarks/{label}/{video}_f{idx:05d}_px.npy   (478, 2) int32
    """
    frames_root = tmp_path / "frames"
    lm_root     = tmp_path / "landmarks"

    rng = np.random.default_rng(0)
    rows = []
    for label, label_int in [("trigger", 1), ("non_trigger", 0)]:
        for v_idx in range(3):
            video = f"VID{v_idx}_{label}"
            for f_idx in range(4):
                stem = f"{video}_f{f_idx:05d}"
                img = (rng.integers(0, 255, (240, 320, 3))).astype(np.uint8)
                (frames_root / label).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frames_root / label / f"{stem}.jpg"), img)

                lm = rng.integers(60, 260, (478, 2)).astype(np.int32)
                (lm_root / label).mkdir(parents=True, exist_ok=True)
                np.save(lm_root / label / f"{stem}_px.npy", lm)
                np.save(lm_root / label / f"{stem}.npy", (lm / 320.0).astype(np.float32))

                rows.append({
                    "stem": stem, "video": video, "label": label,
                    "label_int": label_int, "frame_idx": f_idx,
                })

    return {"frames_root": frames_root, "lm_root": lm_root, "rows": rows}
```

- [ ] **Step 2: Write failing tests for `build_index` and `split_indices`**

```python
# tests/dl_pipeline/test_dataset.py
import numpy as np
import pytest
from src.dl_pipeline.dataset import build_index, split_indices


def test_build_index_pairs_frames_and_landmarks(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    assert len(df) == 24                                      # 3 vids x 4 frames x 2 labels
    assert set(df.columns) >= {"frame_path", "landmarks_path", "label", "video"}
    assert df["frame_path"].apply(lambda p: p.exists()).all()
    assert df["landmarks_path"].apply(lambda p: p.exists()).all()
    assert set(df["label"].unique()) == {0, 1}


def test_build_index_extracts_video_id_from_stem(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    assert (df.groupby("video").size() == 4).all()


def test_split_indices_groups_by_video_with_seed_42(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    train_idx, test_idx = split_indices(df, test_size=0.25, random_state=42)
    train_videos = set(df.iloc[train_idx]["video"])
    test_videos  = set(df.iloc[test_idx]["video"])
    assert train_videos.isdisjoint(test_videos)
    assert len(train_idx) > 0 and len(test_idx) > 0


def test_split_indices_is_deterministic(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    a = split_indices(df, random_state=42)
    b = split_indices(df, random_state=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
```

- [ ] **Step 3: Run tests to verify they fail (module doesn't exist yet)**

```bash
cd /Users/matthewmaingot/CV_Group_Project/Driver-Fatigue-Detection
pytest tests/dl_pipeline/test_dataset.py -v
```

Expected: ImportError / ModuleNotFoundError on `src.dl_pipeline.dataset`.

- [ ] **Step 4: Implement `build_index` and `split_indices`**

```python
# src/dl_pipeline/dataset.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_dataset.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dl_pipeline/dataset.py tests/dl_pipeline/conftest.py tests/dl_pipeline/test_dataset.py
git commit -m "feat(dl_pipeline): add dataset index + group-aware split (seed 42)"
```

---

## Task 3: dataset.py — FrameDataset (face crop)

**Files:**
- Modify: `src/dl_pipeline/dataset.py` (append)
- Modify: `tests/dl_pipeline/test_dataset.py` (append)

- [ ] **Step 1: Add failing tests for the bbox crop helper and FrameDataset**

Append to `tests/dl_pipeline/test_dataset.py`:

```python
import torch
from src.dl_pipeline.dataset import face_crop_from_landmarks, FrameDataset


def test_face_crop_returns_square_bgr_within_bounds():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:300, 200:500] = 200
    lm = np.zeros((478, 2), dtype=np.int32)
    lm[:468] = np.random.default_rng(0).integers(low=[200, 100], high=[500, 300], size=(468, 2))
    crop = face_crop_from_landmarks(img, lm, out_size=224)
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_face_crop_handles_degenerate_landmarks():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = np.zeros((478, 2), dtype=np.int32)
    crop = face_crop_from_landmarks(img, lm, out_size=224)
    assert crop is not None
    assert crop.shape == (224, 224, 3)


def test_frame_dataset_yields_tensor_and_label(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    ds = FrameDataset(df, train=False)
    sample = ds[0]
    assert "image" in sample and "label" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["label"] in (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_dataset.py -k "face_crop or frame_dataset" -v
```

Expected: ImportError on `face_crop_from_landmarks` / `FrameDataset`.

- [ ] **Step 3: Implement `face_crop_from_landmarks` and `FrameDataset`**

Append to `src/dl_pipeline/dataset.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_dataset.py -v
```

Expected: all dataset tests pass (7 total).

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/dataset.py tests/dl_pipeline/test_dataset.py
git commit -m "feat(dl_pipeline): add FrameDataset with face crop + ImageNet transforms"
```

---

## Task 4: dataset.py — WindowDataset

**Files:**
- Modify: `src/dl_pipeline/dataset.py` (append)
- Modify: `tests/dl_pipeline/test_dataset.py` (append)

- [ ] **Step 1: Add failing tests for `build_windows` and `WindowDataset`**

Append to `tests/dl_pipeline/test_dataset.py`:

```python
from src.dl_pipeline.dataset import build_windows, WindowDataset


def test_build_windows_emits_correct_shape(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    windows = build_windows(df, window=3, stride=1)
    assert all(len(w["frame_indices"]) == 3 for w in windows)
    for w in windows:
        last_label = df.iloc[w["frame_indices"][-1]]["label"]
        assert w["label"] == last_label


def test_build_windows_does_not_cross_videos(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    windows = build_windows(df, window=3, stride=1)
    for w in windows:
        videos = df.iloc[w["frame_indices"]]["video"].unique()
        assert len(videos) == 1, "Window must not span multiple videos"


def test_window_dataset_returns_feature_sequence_when_features_provided(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    fake_features = np.random.default_rng(0).standard_normal((len(df), 576)).astype(np.float32)
    windows = build_windows(df, window=3, stride=1)
    ds = WindowDataset(df, windows, features=fake_features)
    sample = ds[0]
    assert sample["features"].shape == (3, 576)
    assert sample["features"].dtype == torch.float32
    assert sample["label"] in (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_dataset.py -k "windows or window_dataset" -v
```

- [ ] **Step 3: Implement `build_windows` and `WindowDataset`**

Append to `src/dl_pipeline/dataset.py`:

```python
def build_windows(
    df: pd.DataFrame,
    window: int = config.WINDOW_SIZE,
    stride: int = config.WINDOW_STRIDE_TRAIN,
) -> list[dict]:
    """Generate sliding windows of `window` consecutive frames, per video.

    Caller should pre-sort by (video, frame_path) for strict temporal order.
    Each window: {"frame_indices": [int, ...], "label": int (last frame), "video": str}
    """
    out = []
    for video, sub in df.groupby("video", sort=False):
        idx = sub.index.to_numpy()
        if len(idx) < window:
            continue
        for start in range(0, len(idx) - window + 1, stride):
            win_idx = idx[start:start + window].tolist()
            label = int(df.iloc[win_idx[-1]]["label"])
            out.append({"frame_indices": win_idx, "label": label, "video": video})
    return out


class WindowDataset(Dataset):
    """Window dataset: serves either pre-extracted features or raw face crops.

    If `features` is given (N, F) it's used directly (fast path for LSTM training
    after the CNN has been frozen and cached). Otherwise crops are loaded on the fly.
    """

    def __init__(self, df, windows, features=None, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.windows = windows
        self.features = features
        self.transform = _build_transform(train)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        if self.features is not None:
            seq = self.features[w["frame_indices"]]
            return {
                "features": torch.from_numpy(seq).float(),
                "label": w["label"],
            }
        imgs = []
        for fi in w["frame_indices"]:
            row = self.df.iloc[fi]
            frame_bgr = cv2.imread(str(row["frame_path"]))
            landmarks = np.load(str(row["landmarks_path"]))
            crop_bgr = face_crop_from_landmarks(frame_bgr, landmarks)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            imgs.append(self.transform(crop_rgb))
        return {"images": torch.stack(imgs), "label": w["label"]}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_dataset.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/dataset.py tests/dl_pipeline/test_dataset.py
git commit -m "feat(dl_pipeline): add WindowDataset with sliding-window per-video grouping"
```

---

## Task 5: models.py — build_cnn

**Files:**
- Create: `src/dl_pipeline/models.py`
- Create: `tests/dl_pipeline/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/dl_pipeline/test_models.py
import torch
from src.dl_pipeline.models import build_cnn, CNNFeatureExtractor


def test_build_cnn_outputs_2_class_logits():
    model = build_cnn(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)


def test_cnn_feature_extractor_returns_576_dim():
    extractor = CNNFeatureExtractor(build_cnn(num_classes=2))
    x = torch.randn(2, 3, 224, 224)
    feats = extractor(x)
    assert feats.shape == (2, 576), f"Expected (2, 576), got {feats.shape}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_models.py -v
```

- [ ] **Step 3: Implement `build_cnn` and `CNNFeatureExtractor`**

```python
# src/dl_pipeline/models.py
"""PyTorch model definitions: per-frame CNN and CNN-LSTM."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from . import config


def build_cnn(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """ImageNet-pretrained MobileNetV3-Small with a fresh `num_classes` head."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features  # 1024
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class CNNFeatureExtractor(nn.Module):
    """Extracts the 576-dim pooled feature vector from MobileNetV3-Small."""

    def __init__(self, cnn: nn.Module):
        super().__init__()
        self.features = cnn.features
        self.avgpool = cnn.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 576)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_models.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/models.py tests/dl_pipeline/test_models.py
git commit -m "feat(dl_pipeline): add MobileNetV3-Small CNN + 576-dim feature extractor"
```

---

## Task 6: models.py — build_cnn_lstm

**Files:**
- Modify: `src/dl_pipeline/models.py` (append)
- Modify: `tests/dl_pipeline/test_models.py` (append)

- [ ] **Step 1: Add failing tests for the LSTM head**

Append to `tests/dl_pipeline/test_models.py`:

```python
from src.dl_pipeline.models import build_cnn_lstm, CNNLSTMHead


def test_cnn_lstm_head_processes_window_to_logits():
    head = CNNLSTMHead(in_dim=576, hidden=128, num_classes=2)
    x = torch.randn(4, 16, 576)
    y = head(x)
    assert y.shape == (4, 2)


def test_build_cnn_lstm_processes_image_window_end_to_end():
    model = build_cnn_lstm(num_classes=2)
    x = torch.randn(2, 4, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_models.py -k "lstm" -v
```

- [ ] **Step 3: Implement `CNNLSTMHead`, `CNNLSTM`, and `build_cnn_lstm`**

Append to `src/dl_pipeline/models.py`:

```python
class CNNLSTMHead(nn.Module):
    """BiLSTM over a sequence of CNN features -> many-to-one classification.

    Hidden=128 -> 256 after concat -> 2-class head.
    """

    def __init__(self, in_dim: int = 576, hidden: int = config.LSTM_HIDDEN,
                 dropout: float = config.LSTM_DROPOUT, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # (B, T, 2H)
        last = out[:, -1, :]          # many-to-one
        return self.classifier(self.dropout(last))


class CNNLSTM(nn.Module):
    """End-to-end CNN-LSTM for inference: applies CNN per frame, then BiLSTM."""

    def __init__(self, extractor: CNNFeatureExtractor, head: CNNLSTMHead):
        super().__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.extractor(x).view(b, t, -1)
        return self.head(feats)


def build_cnn_lstm(num_classes: int = config.NUM_CLASSES, cnn: nn.Module | None = None) -> nn.Module:
    """Build a fresh CNN+LSTM. Pass an existing `cnn` to reuse fine-tuned weights."""
    if cnn is None:
        cnn = build_cnn(num_classes=num_classes)
    extractor = CNNFeatureExtractor(cnn)
    head = CNNLSTMHead(in_dim=576, num_classes=num_classes)
    return CNNLSTM(extractor, head)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_models.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/models.py tests/dl_pipeline/test_models.py
git commit -m "feat(dl_pipeline): add BiLSTM head and end-to-end CNN-LSTM model"
```

---

## Task 7: train_eval.py — metrics + plots

**Files:**
- Create: `src/dl_pipeline/train_eval.py`
- Create: `tests/dl_pipeline/test_train_eval.py`

- [ ] **Step 1: Write failing test for `compute_metrics`**

```python
# tests/dl_pipeline/test_train_eval.py
import numpy as np
from src.dl_pipeline.train_eval import compute_metrics


def test_compute_metrics_returns_expected_keys_and_perfect_score():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.05, 0.95])
    m = compute_metrics(y_true, y_pred, y_proba)
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        assert k in m
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert 0.99 <= m["auc"] <= 1.0


def test_compute_metrics_handles_imbalance():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.1, 0.3, 0.4, 0.9])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 5 / 6
    assert m["recall"] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_train_eval.py -v
```

- [ ] **Step 3: Implement `compute_metrics` and plot helpers**

```python
# src/dl_pipeline/train_eval.py
"""Training, evaluation, metrics, and feature caching for the DL pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Standard binary classification metrics. AUC uses positive-class probability."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":       float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_confusion_plot(cm: list[list[int]], out_path: Path, title: str) -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["non_trigger", "trigger"])
    ax.set_yticklabels(["non_trigger", "trigger"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, arr[i, j], ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_roc_plot(curves: dict, out_path: Path) -> None:
    """curves: {model_name: (fpr, tpr, auc)}"""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - DL pipeline")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_train_eval.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/train_eval.py tests/dl_pipeline/test_train_eval.py
git commit -m "feat(dl_pipeline): add metrics + confusion/ROC plot helpers"
```

---

## Task 8: train_eval.py — train_cnn loop

**Files:**
- Modify: `src/dl_pipeline/train_eval.py` (append)
- Modify: `tests/dl_pipeline/test_train_eval.py` (append)

- [ ] **Step 1: Add a smoke test that trains for 1 step and asserts loss is finite**

Append to `tests/dl_pipeline/test_train_eval.py`:

```python
import torch
from torch.utils.data import DataLoader
from src.dl_pipeline.dataset import build_index, FrameDataset
from src.dl_pipeline.models import build_cnn
from src.dl_pipeline.train_eval import train_one_epoch, evaluate_cnn


def test_train_one_epoch_runs_and_returns_finite_loss(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    ds = FrameDataset(df, train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = build_cnn(num_classes=2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    avg_loss = train_one_epoch(model, loader, optim, loss_fn, device="cpu")
    assert np.isfinite(avg_loss)


def test_evaluate_cnn_returns_metrics_dict(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    ds = FrameDataset(df, train=False)
    loader = DataLoader(ds, batch_size=4)
    model = build_cnn(num_classes=2)
    metrics, y_true, y_pred, y_proba = evaluate_cnn(model, loader, device="cpu")
    for k in ("accuracy", "f1", "auc"):
        assert k in metrics
    assert len(y_true) == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_train_eval.py -k "train_one_epoch or evaluate_cnn" -v
```

- [ ] **Step 3: Implement training and evaluation loops**

Append to `src/dl_pipeline/train_eval.py`:

```python
def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer,
                    loss_fn: nn.Module, device: str) -> float:
    model.train(True)
    total, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate_cnn(model: nn.Module, loader: DataLoader, device: str
                 ):
    model.train(False)
    ys, preds, probs = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        preds.append(logits.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true  = np.concatenate(ys)
    y_pred  = np.concatenate(preds)
    y_proba = np.concatenate(probs)
    return compute_metrics(y_true, y_pred, y_proba), y_true, y_pred, y_proba


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = config.CNN_EPOCHS,
    head_lr: float = config.CNN_HEAD_LR,
    bb_lr: float = config.CNN_BB_LR,
    weight_decay: float = config.CNN_WEIGHT_DECAY,
    class_weights: torch.Tensor | None = None,
    device: str = "cpu",
    out_path: Path = config.OUTPUTS_DIR / "cnn_mobilenetv3.pt",
    log_dir: Path | None = None,
) -> dict:
    """Fine-tune MobileNetV3-Small. Unfreezes last 2 feature blocks at lower LR.

    Saves best (val F1) checkpoint to `out_path`. Returns final-epoch metrics dict.
    """
    from .models import build_cnn

    model = build_cnn(num_classes=config.NUM_CLASSES).to(device)

    for p in model.parameters():
        p.requires_grad = False
    backbone_params = []
    for blk in list(model.features.children())[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
            backbone_params.append(p)
    head_params = [p for p in model.classifier.parameters()]
    for p in head_params:
        p.requires_grad = True

    optim = torch.optim.AdamW(
        [{"params": head_params, "lr": head_lr},
         {"params": backbone_params, "lr": bb_lr}],
        weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_f1, best_metrics = -1.0, {}
    for epoch in range(epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
        sched.step()
        val_metrics, *_ = evaluate_cnn(model, val_loader, device)
        print(f"[CNN] epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"({time.time()-t0:.1f}s)")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "cnn_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    return best_metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_train_eval.py -v
```

Note: this downloads MobileNetV3-Small weights on first run (~10 MB).

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/train_eval.py tests/dl_pipeline/test_train_eval.py
git commit -m "feat(dl_pipeline): add CNN training loop with cosine LR + best-F1 checkpointing"
```

---

## Task 9: train_eval.py — feature caching

**Files:**
- Modify: `src/dl_pipeline/train_eval.py` (append)
- Modify: `tests/dl_pipeline/test_train_eval.py` (append)

- [ ] **Step 1: Add failing test for `cache_features`**

Append to `tests/dl_pipeline/test_train_eval.py`:

```python
from src.dl_pipeline.train_eval import cache_features
from src.dl_pipeline.models import build_cnn, CNNFeatureExtractor


def test_cache_features_writes_one_array_per_video(fake_dataset, tmp_path):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    extractor = CNNFeatureExtractor(build_cnn(num_classes=2))
    out_dir = tmp_path / "feature_cache"
    feats_arr = cache_features(df, extractor, out_dir=out_dir, device="cpu", batch_size=4)
    assert feats_arr.shape == (len(df), 576)
    assert len(list(out_dir.glob("*.npy"))) == df["video"].nunique()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_train_eval.py -k cache_features -v
```

- [ ] **Step 3: Implement `cache_features`**

Append to `src/dl_pipeline/train_eval.py`:

```python
@torch.no_grad()
def cache_features(
    df,
    extractor: nn.Module,
    out_dir: Path,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 576-dim CNN features for every frame in `df`, cached per-video."""
    from .dataset import FrameDataset
    extractor.train(False)
    extractor.to(device)
    ds = FrameDataset(df, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    for batch in loader:
        x = batch["image"].to(device)
        f = extractor(x).cpu().numpy()
        feats.append(f)
    feats_arr = np.concatenate(feats, axis=0).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    for video, sub in df.groupby("video", sort=False):
        idx = sub.index.to_numpy()
        np.save(out_dir / f"{video}.npy", feats_arr[idx])
    return feats_arr
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_train_eval.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/train_eval.py tests/dl_pipeline/test_train_eval.py
git commit -m "feat(dl_pipeline): cache CNN features per-video for fast LSTM training"
```

---

## Task 10: train_eval.py — train_cnn_lstm loop

**Files:**
- Modify: `src/dl_pipeline/train_eval.py` (append)
- Modify: `tests/dl_pipeline/test_train_eval.py` (append)

- [ ] **Step 1: Add smoke test for `train_lstm` and `evaluate_lstm`**

Append to `tests/dl_pipeline/test_train_eval.py`:

```python
from src.dl_pipeline.dataset import build_windows, WindowDataset
from src.dl_pipeline.train_eval import train_lstm, evaluate_lstm
from src.dl_pipeline.models import CNNLSTMHead


def test_train_lstm_runs_one_step(fake_dataset, tmp_path):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    feats = np.random.default_rng(0).standard_normal((len(df), 576)).astype(np.float32)
    windows = build_windows(df, window=3, stride=1)
    train_ds = WindowDataset(df, windows, features=feats)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(train_ds, batch_size=4)
    metrics = train_lstm(
        train_loader, val_loader,
        epochs=1, in_dim=576, hidden=8, lr=1e-3, device="cpu",
        out_path=tmp_path / "lstm.pt",
    )
    assert "accuracy" in metrics


def test_evaluate_lstm_returns_metrics(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    feats = np.random.default_rng(0).standard_normal((len(df), 576)).astype(np.float32)
    windows = build_windows(df, window=3, stride=1)
    ds = WindowDataset(df, windows, features=feats)
    loader = DataLoader(ds, batch_size=4)
    head = CNNLSTMHead(in_dim=576, hidden=8, num_classes=2)
    metrics, *_ = evaluate_lstm(head, loader, device="cpu")
    assert "f1" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_train_eval.py -k "train_lstm or evaluate_lstm" -v
```

- [ ] **Step 3: Implement `train_lstm` and `evaluate_lstm`**

Append to `src/dl_pipeline/train_eval.py`:

```python
@torch.no_grad()
def evaluate_lstm(head: nn.Module, loader: DataLoader, device: str):
    head.train(False)
    head.to(device)
    ys, preds, probs = [], [], []
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)
        logits = head(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        preds.append(logits.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true  = np.concatenate(ys)
    y_pred  = np.concatenate(preds)
    y_proba = np.concatenate(probs)
    return compute_metrics(y_true, y_pred, y_proba), y_true, y_pred, y_proba


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = config.LSTM_EPOCHS,
    in_dim: int = 576,
    hidden: int = config.LSTM_HIDDEN,
    lr: float = config.LSTM_LR,
    class_weights: torch.Tensor | None = None,
    device: str = "cpu",
    out_path: Path = config.OUTPUTS_DIR / "cnn_lstm.pt",
    log_dir: Path | None = None,
) -> dict:
    """Train BiLSTM head on cached CNN features. CNN stays frozen."""
    from .models import CNNLSTMHead

    head = CNNLSTMHead(in_dim=in_dim, hidden=hidden, num_classes=config.NUM_CLASSES).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_f1, best_metrics = -1.0, {}
    for epoch in range(epochs):
        t0 = time.time()
        head.train(True)
        total, n = 0.0, 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            optim.zero_grad()
            logits = head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
        avg_loss = total / max(1, n)
        val_metrics, *_ = evaluate_lstm(head, val_loader, device)
        print(f"[LSTM] epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"({time.time()-t0:.1f}s)")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), out_path)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "lstm_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    return best_metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_train_eval.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/train_eval.py tests/dl_pipeline/test_train_eval.py
git commit -m "feat(dl_pipeline): add BiLSTM head training on cached features"
```

---

## Task 11: predict.py — DLPredictor

**Files:**
- Create: `src/dl_pipeline/predict.py`
- Create: `tests/dl_pipeline/test_predict.py`

- [ ] **Step 1: Write failing tests for the predictor API**

```python
# tests/dl_pipeline/test_predict.py
import numpy as np
import torch
from src.dl_pipeline.predict import DLPredictor
from src.dl_pipeline.models import build_cnn, CNNLSTMHead


def _save_dummy_ckpts(tmp_path):
    cnn_path = tmp_path / "cnn.pt"
    lstm_path = tmp_path / "lstm.pt"
    torch.save(build_cnn(num_classes=2).state_dict(), cnn_path)
    torch.save(CNNLSTMHead(in_dim=576, hidden=8, num_classes=2).state_dict(), lstm_path)
    return cnn_path, lstm_path


def test_predict_frame_returns_label_proba_model(tmp_path):
    cnn_path, _ = _save_dummy_ckpts(tmp_path)
    pred = DLPredictor(cnn_ckpt=cnn_path, device="cpu")
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    lm = np.random.default_rng(0).integers(60, 260, (478, 2), dtype=np.int32)
    out = pred.predict_frame(img, lm)
    assert set(out.keys()) >= {"label", "proba", "model"}
    assert out["label"] in (0, 1)
    assert 0.0 <= out["proba"] <= 1.0
    assert out["model"] == "cnn"


def test_predict_window_returns_lstm_model(tmp_path):
    cnn_path, lstm_path = _save_dummy_ckpts(tmp_path)
    pred = DLPredictor(cnn_ckpt=cnn_path, lstm_ckpt=lstm_path, lstm_hidden=8, device="cpu")
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(4)]
    landmarks = [rng.integers(60, 260, (478, 2), dtype=np.int32) for _ in range(4)]
    out = pred.predict_window(frames, landmarks)
    assert out["model"] == "cnn_lstm"
    assert out["label"] in (0, 1)
    assert 0.0 <= out["proba"] <= 1.0


def test_predict_window_without_lstm_raises(tmp_path):
    cnn_path, _ = _save_dummy_ckpts(tmp_path)
    pred = DLPredictor(cnn_ckpt=cnn_path, device="cpu")
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(4)]
    landmarks = [rng.integers(60, 260, (478, 2), dtype=np.int32) for _ in range(4)]
    try:
        pred.predict_window(frames, landmarks)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when lstm_ckpt not provided")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dl_pipeline/test_predict.py -v
```

- [ ] **Step 3: Implement `DLPredictor`**

```python
# src/dl_pipeline/predict.py
"""Inference API consumed by Member 5's fusion layer.

Mirrors the shape of the classical SVM's predict / predict_proba so the
fusion layer can treat the DL branch identically to the classical branch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torchvision import transforms

from . import config
from .dataset import face_crop_from_landmarks
from .models import build_cnn, CNNFeatureExtractor, CNNLSTMHead


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _eval_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class DLPredictor:
    """Loads the trained CNN (and optional LSTM head) for inference."""

    def __init__(
        self,
        cnn_ckpt: Path,
        lstm_ckpt: Path | None = None,
        lstm_hidden: int = config.LSTM_HIDDEN,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.cnn = build_cnn(num_classes=config.NUM_CLASSES)
        self.cnn.load_state_dict(torch.load(str(cnn_ckpt), map_location=self.device))
        self.cnn.train(False)
        self.cnn.to(self.device)
        self.extractor = CNNFeatureExtractor(self.cnn)
        self.extractor.train(False)
        self.extractor.to(self.device)
        self.lstm: CNNLSTMHead | None = None
        if lstm_ckpt is not None:
            self.lstm = CNNLSTMHead(in_dim=576, hidden=lstm_hidden, num_classes=config.NUM_CLASSES)
            self.lstm.load_state_dict(torch.load(str(lstm_ckpt), map_location=self.device))
            self.lstm.train(False)
            self.lstm.to(self.device)
        self.transform = _eval_transform()

    def _to_tensor(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> torch.Tensor:
        crop = face_crop_from_landmarks(frame_bgr, landmarks_px)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> dict:
        x = self._to_tensor(frame_bgr, landmarks_px).unsqueeze(0).to(self.device)
        logits = self.cnn(x)
        proba = torch.softmax(logits, dim=1)[0, 1].item()
        label = int(logits.argmax(dim=1).item())
        return {"label": label, "proba": float(proba), "model": "cnn"}

    @torch.no_grad()
    def predict_window(
        self,
        frames_bgr: Sequence[np.ndarray],
        landmarks_px_seq: Sequence[np.ndarray],
    ) -> dict:
        if self.lstm is None:
            raise RuntimeError("DLPredictor was constructed without an LSTM checkpoint; "
                               "pass lstm_ckpt= to use predict_window().")
        if len(frames_bgr) != len(landmarks_px_seq):
            raise ValueError("frames_bgr and landmarks_px_seq must be the same length")
        tensors = [self._to_tensor(f, lm) for f, lm in zip(frames_bgr, landmarks_px_seq)]
        x = torch.stack(tensors).to(self.device)
        feats = self.extractor(x).unsqueeze(0)        # (1, T, 576)
        logits = self.lstm(feats)
        proba = torch.softmax(logits, dim=1)[0, 1].item()
        label = int(logits.argmax(dim=1).item())
        return {"label": label, "proba": float(proba), "model": "cnn_lstm"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dl_pipeline/test_predict.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dl_pipeline/predict.py tests/dl_pipeline/test_predict.py
git commit -m "feat(dl_pipeline): add DLPredictor with predict_frame and predict_window APIs"
```

---

## Task 12: run_dl_pipeline.py — CLI

**Files:**
- Create: `src/dl_pipeline/run_dl_pipeline.py`

- [ ] **Step 1: Implement the CLI**

```python
# src/dl_pipeline/run_dl_pipeline.py
"""CLI entry point for the DL pipeline.

Modes:
  train_cnn       - fine-tune MobileNetV3-Small on our captured data
  cache_features  - write per-video 576-dim feature .npy files
  train_lstm      - train the BiLSTM head on cached features
  evaluate        - load both checkpoints, dump metrics + plots

Examples:
  python -m src.dl_pipeline.run_dl_pipeline train_cnn --epochs 20
  python -m src.dl_pipeline.run_dl_pipeline cache_features
  python -m src.dl_pipeline.run_dl_pipeline train_lstm --epochs 25
  python -m src.dl_pipeline.run_dl_pipeline evaluate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import (
    build_index, split_indices, FrameDataset, WindowDataset, build_windows,
)
from .models import build_cnn, CNNFeatureExtractor, CNNLSTMHead
from .train_eval import (
    train_cnn, cache_features, train_lstm,
    evaluate_cnn, evaluate_lstm,
    save_confusion_plot, save_roc_plot,
)
from sklearn.metrics import roc_curve


def _seed_everything(seed: int = config.GLOBAL_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_loaders_cnn(df, train_idx, test_idx, batch_size: int):
    train_ds = FrameDataset(df.iloc[train_idx].reset_index(drop=True), train=True)
    val_ds   = FrameDataset(df.iloc[test_idx].reset_index(drop=True),  train=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def _class_weights(df) -> torch.Tensor:
    counts = np.bincount(df["label"].values.astype(int), minlength=2).astype(np.float32)
    inv = counts.sum() / (2.0 * counts.clip(min=1))
    return torch.tensor(inv, dtype=torch.float32)


def cmd_train_cnn(args):
    _seed_everything()
    df = build_index(config.DATA_FRAMES, config.DATA_LANDMARKS)
    train_idx, test_idx = split_indices(df)
    train_loader, val_loader = _build_loaders_cnn(df, train_idx, test_idx, args.batch_size)
    weights = _class_weights(df.iloc[train_idx]) if args.class_weights else None
    log_dir = config.OUTPUTS_DIR / "runs" / time.strftime("%Y%m%d-%H%M%S")
    train_cnn(
        train_loader, val_loader,
        epochs=args.epochs, device=_device(args.device),
        class_weights=weights, log_dir=log_dir,
    )


def cmd_cache_features(args):
    _seed_everything()
    df = build_index(config.DATA_FRAMES, config.DATA_LANDMARKS).sort_values(
        ["video", "frame_path"]).reset_index(drop=True)
    cnn = build_cnn(num_classes=config.NUM_CLASSES)
    cnn.load_state_dict(torch.load(str(config.OUTPUTS_DIR / "cnn_mobilenetv3.pt"),
                                   map_location="cpu"))
    extractor = CNNFeatureExtractor(cnn)
    cache_features(df, extractor, out_dir=config.FEATURE_CACHE,
                   device=_device(args.device), batch_size=args.batch_size)


def _load_features_in_df_order(df) -> np.ndarray:
    feats = np.zeros((len(df), 576), dtype=np.float32)
    for video, sub in df.groupby("video", sort=False):
        arr = np.load(config.FEATURE_CACHE / f"{video}.npy")
        feats[sub.index.to_numpy()] = arr
    return feats


def cmd_train_lstm(args):
    _seed_everything()
    df = build_index(config.DATA_FRAMES, config.DATA_LANDMARKS).sort_values(
        ["video", "frame_path"]).reset_index(drop=True)
    train_idx, test_idx = split_indices(df)
    feats = _load_features_in_df_order(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    train_feats = feats[train_idx]
    test_feats  = feats[test_idx]
    train_windows = build_windows(train_df, window=config.WINDOW_SIZE,
                                  stride=config.WINDOW_STRIDE_TRAIN)
    test_windows  = build_windows(test_df,  window=config.WINDOW_SIZE,
                                  stride=config.WINDOW_STRIDE_EVAL)
    train_loader = DataLoader(WindowDataset(train_df, train_windows, features=train_feats),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(test_df, test_windows, features=test_feats),
                              batch_size=args.batch_size)
    weights = _class_weights(train_df) if args.class_weights else None
    log_dir = config.OUTPUTS_DIR / "runs" / time.strftime("%Y%m%d-%H%M%S")
    train_lstm(
        train_loader, val_loader,
        epochs=args.epochs, device=_device(args.device),
        class_weights=weights, log_dir=log_dir,
    )


def cmd_evaluate(args):
    _seed_everything()
    device = _device(args.device)
    df = build_index(config.DATA_FRAMES, config.DATA_LANDMARKS).sort_values(
        ["video", "frame_path"]).reset_index(drop=True)
    train_idx, test_idx = split_indices(df)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # CNN evaluation
    cnn = build_cnn(num_classes=config.NUM_CLASSES)
    cnn.load_state_dict(torch.load(str(config.OUTPUTS_DIR / "cnn_mobilenetv3.pt"),
                                   map_location=device))
    cnn.to(device)
    cnn_loader = DataLoader(FrameDataset(test_df, train=False),
                            batch_size=args.batch_size, num_workers=2)
    cnn_metrics, _, _, cnn_proba = evaluate_cnn(cnn, cnn_loader, device=device)

    # LSTM evaluation (uses cached features in test order)
    feats = _load_features_in_df_order(df)
    test_feats = feats[test_idx]
    test_windows = build_windows(test_df, window=config.WINDOW_SIZE,
                                 stride=config.WINDOW_STRIDE_EVAL)
    lstm_loader = DataLoader(WindowDataset(test_df, test_windows, features=test_feats),
                             batch_size=args.batch_size)
    head = CNNLSTMHead(in_dim=576, hidden=config.LSTM_HIDDEN, num_classes=config.NUM_CLASSES)
    head.load_state_dict(torch.load(str(config.OUTPUTS_DIR / "cnn_lstm.pt"),
                                    map_location=device))
    head.to(device)
    lstm_metrics, lstm_y, lstm_pred, lstm_proba = evaluate_lstm(head, lstm_loader, device=device)

    save_confusion_plot(cnn_metrics["confusion_matrix"],
                        config.OUTPUTS_DIR / "confusion_cnn.png", "CNN")
    save_confusion_plot(lstm_metrics["confusion_matrix"],
                        config.OUTPUTS_DIR / "confusion_cnn_lstm.png", "CNN-LSTM")

    cnn_y = test_df["label"].values.astype(int)
    fpr_c, tpr_c, _ = roc_curve(cnn_y, cnn_proba)
    fpr_l, tpr_l, _ = roc_curve(lstm_y, lstm_proba)
    save_roc_plot({
        "CNN":      (fpr_c, tpr_c, cnn_metrics["auc"]),
        "CNN-LSTM": (fpr_l, tpr_l, lstm_metrics["auc"]),
    }, config.OUTPUTS_DIR / "roc_curves.png")

    out = {"cnn": cnn_metrics, "cnn_lstm": lstm_metrics}
    (config.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    common.add_argument("--batch-size", type=int, default=64)
    common.add_argument("--class-weights", action="store_true")

    p_cnn = sub.add_parser("train_cnn", parents=[common])
    p_cnn.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    p_cnn.set_defaults(func=cmd_train_cnn)

    p_cache = sub.add_parser("cache_features", parents=[common])
    p_cache.set_defaults(func=cmd_cache_features)

    p_lstm = sub.add_parser("train_lstm", parents=[common])
    p_lstm.add_argument("--epochs", type=int, default=config.LSTM_EPOCHS)
    p_lstm.set_defaults(func=cmd_train_lstm)

    p_eval = sub.add_parser("evaluate", parents=[common])
    p_eval.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses correctly**

```bash
cd /Users/matthewmaingot/CV_Group_Project/Driver-Fatigue-Detection
python -m src.dl_pipeline.run_dl_pipeline --help
python -m src.dl_pipeline.run_dl_pipeline train_cnn --help
```

Expected: usage strings printed, no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/dl_pipeline/run_dl_pipeline.py
git commit -m "feat(dl_pipeline): add CLI for train_cnn / cache_features / train_lstm / evaluate"
```

---

## Task 13: End-to-end smoke run on real data

This task runs once on the user's machine (Colab/Kaggle/local), not in pytest. It validates the pipeline end-to-end on real captured data.

- [ ] **Step 1: Install dependencies**

```bash
cd /Users/matthewmaingot/CV_Group_Project/Driver-Fatigue-Detection
pip install -r src/dl_pipeline/requirements_dl_pipeline.txt
```

- [ ] **Step 2: Run the full pytest suite**

```bash
pytest tests/dl_pipeline/ -v
```

Expected: all tests pass. Fix any failures before proceeding to real-data training.

- [ ] **Step 3: Train CNN for 1 epoch as a smoke test**

```bash
python -m src.dl_pipeline.run_dl_pipeline train_cnn --epochs 1 --batch-size 32 --device auto --class-weights
```

Expected: prints `[CNN] epoch 1/1  loss=...  val_f1=...  val_acc=...`. Loss should be finite, val_f1 above 0.5 even after 1 epoch.

- [ ] **Step 4: Cache features end-to-end**

```bash
python -m src.dl_pipeline.run_dl_pipeline cache_features --device auto --batch-size 64
```

Expected: `outputs/dl_pipeline/feature_cache/*.npy` created, one per source video.

- [ ] **Step 5: Train LSTM for 2 epochs**

```bash
python -m src.dl_pipeline.run_dl_pipeline train_lstm --epochs 2 --batch-size 64 --device auto --class-weights
```

Expected: prints two `[LSTM] epoch ...` lines, finite loss, val_f1 above 0.5.

- [ ] **Step 6: Run full evaluation and inspect outputs**

```bash
python -m src.dl_pipeline.run_dl_pipeline evaluate --device auto
ls outputs/dl_pipeline/
cat outputs/dl_pipeline/metrics.json
```

Expected files: `cnn_mobilenetv3.pt`, `cnn_lstm.pt`, `confusion_cnn.png`, `confusion_cnn_lstm.png`, `roc_curves.png`, `metrics.json`. JSON contains accuracy/precision/recall/f1/auc for both models.

- [ ] **Step 7: Quick latency check**

Open a Python REPL and time inference:

```python
import os, time
import cv2, numpy as np
from pathlib import Path
from src.dl_pipeline.predict import DLPredictor
from src.dl_pipeline import config

pred = DLPredictor(cnn_ckpt=config.OUTPUTS_DIR / "cnn_mobilenetv3.pt", device="cpu")
frames_dir = Path("data/frames/trigger")
lm_dir = Path("data/landmarks/trigger")
sample_jpg = sorted(frames_dir.glob("*.jpg"))[0]
sample_lm  = lm_dir / f"{sample_jpg.stem}_px.npy"
img = cv2.imread(str(sample_jpg))
lm  = np.load(sample_lm)
for _ in range(5): pred.predict_frame(img, lm)             # warmup
t0 = time.perf_counter()
for _ in range(100): pred.predict_frame(img, lm)
print(f"avg: {(time.perf_counter() - t0) * 10:.2f} ms/frame")
```

Expected: under ~30 ms/frame on CPU. If slower, document in the report.

- [ ] **Step 8: Add gitignore entries and commit small artifacts**

```bash
echo "outputs/dl_pipeline/*.pt" >> .gitignore
echo "outputs/dl_pipeline/feature_cache/" >> .gitignore
echo "outputs/dl_pipeline/runs/" >> .gitignore
git add .gitignore
git add outputs/dl_pipeline/metrics.json outputs/dl_pipeline/*.png
git commit -m "feat(dl_pipeline): end-to-end smoke run with metrics + ROC + confusion plots"
```

---

## Verification Summary

After Task 13:

- pytest tests/dl_pipeline/ -v passes
- outputs/dl_pipeline/cnn_mobilenetv3.pt exists
- outputs/dl_pipeline/cnn_lstm.pt exists
- outputs/dl_pipeline/metrics.json contains both `cnn` and `cnn_lstm` results
- ROC + confusion plots rendered
- DLPredictor.predict_frame and predict_window work on real frames
- Inference latency under ~30 ms/frame on CPU (real-time-viable)
- Test split matches Member 3's exactly (same `random_state=42`, same videos)

Hand off to Member 5: import path is `from src.dl_pipeline.predict import DLPredictor`. Pass paths to the two checkpoints and the fusion layer is unblocked.

For the technical report, the numbers in `outputs/dl_pipeline/metrics.json` plus Member 3's `outputs/classical_pipeline/metrics.json` make the SVM / RF / kNN / CNN / CNN-LSTM comparison table directly.
