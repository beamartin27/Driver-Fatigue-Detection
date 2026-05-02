# Member 4 — DL Pipeline Handoff

**Branch:** `member4/dl-pipeline` (pushed to origin)
**Plan:** [docs/superpowers/plans/2026-04-21-dl-pipeline.md](plans/2026-04-21-dl-pipeline.md)
**Status:** 13 of 14 tasks done (Tasks 0–12). Task 13 (real-data smoke run on Colab) is left for the user to execute.
**Test status:** `pytest tests/dl_pipeline/ -v` → **24 passed**.

---

## What's done — code complete

| # | Task | Files | Commit |
|---|---|---|---|
| 0 | Scaffold package | `src/dl_pipeline/__init__.py`, `requirements_dl_pipeline.txt`, `tests/dl_pipeline/__init__.py` | `1b743a7d` |
| 1 | config.py | locked seeds, paths, hyperparams | `02b29d45` |
| 2 | dataset index + split | `build_index`, `split_indices`, `pytest.ini`, `conftest.py` | `5ae8ec74` |
| 3 | FrameDataset + face crop | `face_crop_from_landmarks`, `FrameDataset` | `164db709` |
| 4 | WindowDataset | `build_windows`, `WindowDataset` | `185da596` |
| 5 | CNN model | `build_cnn`, `CNNFeatureExtractor` (MobileNetV3-Small + 576-dim extractor) | `39eb168a` |
| 6 | CNN-LSTM model | `CNNLSTMHead`, `CNNLSTM`, `build_cnn_lstm` (BiLSTM, hidden=128) | `58ba5c26` |
| 7 | metrics + plots | `compute_metrics`, `save_confusion_plot`, `save_roc_plot` | `a772a6b0` |
| 8 | CNN training loop | `train_one_epoch`, `evaluate_cnn`, `train_cnn` (cosine LR + best-F1 ckpt) | `8045f983` |
| 9 | feature caching | `cache_features` (576-dim per-video `.npy`) | `2ebeceb0` |
| 10 | LSTM training loop | `train_lstm`, `evaluate_lstm` | `6c82908a` |
| 11 | DLPredictor (Member 5 fusion API) | `src/dl_pipeline/predict.py` — `predict_frame`, `predict_window` | `b904bee0` |
| 12 | CLI entry point | `src/dl_pipeline/run_dl_pipeline.py` (subcommands: `train_cnn / cache_features / train_lstm / evaluate`) | `eccf2e44` |

---

## What's left — Task 13 only (real-data smoke run, ~Colab)

Task 13 trains on the team's actual captured frames in `data/frames/{trigger,non_trigger}/`. Recommended to run on Google Colab (free T4 GPU). On a CPU it would take hours.

### One-time setup on Colab

1. Mount Drive (or upload the repo).
2. Install deps:
   ```bash
   cd /content/Driver-Fatigue-Detection
   pip install -r src/dl_pipeline/requirements_dl_pipeline.txt
   ```
3. Sanity check — full pytest suite:
   ```bash
   pytest tests/dl_pipeline/ -v          # expect 24 passed
   ```

### Run the four CLI stages

```bash
# Stage A — fine-tune MobileNetV3-Small (~10 min on T4 for 20 epochs)
python -m src.dl_pipeline.run_dl_pipeline train_cnn --epochs 20 --batch-size 64 --device auto --class-weights

# Stage B — cache 576-dim features per video
python -m src.dl_pipeline.run_dl_pipeline cache_features --device auto --batch-size 64

# Stage C — train BiLSTM head on cached features (~2 min on T4 for 25 epochs)
python -m src.dl_pipeline.run_dl_pipeline train_lstm --epochs 25 --batch-size 128 --device auto --class-weights

# Stage D — produce metrics.json + confusion + ROC plots
python -m src.dl_pipeline.run_dl_pipeline evaluate --device auto
```

### What gets produced

```
outputs/dl_pipeline/
├── cnn_mobilenetv3.pt          (Model A checkpoint)
├── cnn_lstm.pt                 (Model B checkpoint)
├── feature_cache/<video>.npy   (576-dim per-video features)
├── metrics.json                (Acc/Prec/Rec/F1/AUC for both models)
├── confusion_cnn.png
├── confusion_cnn_lstm.png
├── roc_curves.png
└── runs/<timestamp>/           (per-run logs)
```

### Latency check (real-time-viability for the demo)

In a Python REPL (CPU is fine):
```python
import time, cv2, numpy as np
from pathlib import Path
from src.dl_pipeline.predict import DLPredictor
from src.dl_pipeline import config

pred = DLPredictor(cnn_ckpt=config.OUTPUTS_DIR / "cnn_mobilenetv3.pt", device="cpu")
sample_jpg = sorted(Path("data/frames/trigger").glob("*.jpg"))[0]
sample_lm  = Path("data/landmarks/trigger") / f"{sample_jpg.stem}_px.npy"
img, lm = cv2.imread(str(sample_jpg)), np.load(sample_lm)
for _ in range(5): pred.predict_frame(img, lm)               # warmup
t0 = time.perf_counter()
for _ in range(100): pred.predict_frame(img, lm)
print(f"avg: {(time.perf_counter() - t0) * 10:.2f} ms/frame")  # target: < 30 ms
```

### Commit the artifacts (small files only)

```bash
git add outputs/dl_pipeline/metrics.json outputs/dl_pipeline/*.png
git commit -m "results(dl_pipeline): end-to-end smoke run metrics + plots"
```

The `.gitignore` already excludes the `.pt` checkpoints, `feature_cache/`, and `runs/` (those are too big for git).

---

## Hand-off to Member 5 (fusion layer)

Single import. Mirrors the classical SVM `predict` / `predict_proba` API so no adapter code is needed:

```python
from src.dl_pipeline.predict import DLPredictor

pred = DLPredictor(
    cnn_ckpt="outputs/dl_pipeline/cnn_mobilenetv3.pt",
    lstm_ckpt="outputs/dl_pipeline/cnn_lstm.pt",   # optional; omit for CNN-only
    device="cpu",
)

# Per-frame (low latency, real-time path)
pred.predict_frame(frame_bgr, landmarks_px)
# → {"label": 0|1, "proba": float, "model": "cnn"}

# Windowed (better accuracy, requires a 16-frame buffer)
pred.predict_window(frames_bgr, landmarks_px_seq)
# → {"label": 0|1, "proba": float, "model": "cnn_lstm"}
```

The locked test split (`GroupShuffleSplit(test_size=0.25, random_state=42)`) is identical to Member 3's, so the comparison table for the report drops in cleanly:

| Model | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| SVM (Member 3) | 86.34% | … | … | 0.8893 | … |
| RF (Member 3) | … | … | … | … | … |
| kNN (Member 3) | … | … | … | … | … |
| CNN (Member 4) | from `outputs/dl_pipeline/metrics.json` | | | | |
| CNN-LSTM (Member 4) | from `outputs/dl_pipeline/metrics.json` | | | | |

---

## Open the PR when Task 13 is done

```bash
gh pr create --title "Member 4 — Deep learning pipeline (CNN + CNN-LSTM)" \
  --body "Implements the DL branch per docs/superpowers/plans/2026-04-21-dl-pipeline.md.

Two PyTorch models share a MobileNetV3-Small backbone:
- CNN — per-frame fatigue classifier (fine-tuned from ImageNet on team data)
- CNN-LSTM — BiLSTM over 16-frame windows of cached CNN features

Test split locked to GroupShuffleSplit(test_size=0.25, random_state=42) — identical to Member 3's classical baseline so SVM/RF/kNN/CNN/CNN-LSTM are directly comparable.

Exposes DLPredictor with predict_frame() / predict_window() (mirrors Member 3's SVM API) for Member 5's late-fusion layer.

Tests: 24 unit tests, all passing. End-to-end smoke run + real-data metrics in outputs/dl_pipeline/."
```

---

## Environment notes (one gotcha)

If you start fresh on this machine, you may need to reinstall torch into the conda env that was used:
```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python3.10 -m pip install -r src/dl_pipeline/requirements_dl_pipeline.txt
# If numpy 2.x breaks pandas: pip install 'numpy<2'
```
On Colab/Kaggle this is a non-issue — they ship modern torch + numpy.

---

## Architecture refresher

```
frame (BGR) ─► MediaPipe bbox (Member 2's landmarks_px)
             │
             ▼
        crop + resize 224×224
             │
             ▼
        MobileNetV3-Small (ImageNet → fine-tuned on ours)
             │
             ├───► Linear(576, 2) ──► P(fatigue) per frame    [CNN  — Model A]
             │
             └───► 576-dim feature
                       │
                       ▼
                  16-frame window (4 sec @ 4 fps)
                       │
                       ▼
                  BiLSTM(hidden=128, dropout=0.3)
                       │
                       ▼
                  Linear(256, 2) ──► P(fatigue) for last frame [CNN-LSTM — Model B]
```
