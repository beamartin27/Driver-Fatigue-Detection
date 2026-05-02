# Member 4 — DL Pipeline Handoff

**Branch:** `member4/dl-pipeline` (pushed to origin)
**Plan:** [docs/superpowers/plans/2026-04-21-dl-pipeline.md](plans/2026-04-21-dl-pipeline.md)
**Status:** 8 of 14 tasks done (Tasks 0–7). Tasks 8–13 remain.

---

## What's done

| # | Task | Files created | Commit |
|---|---|---|---|
| 0 | Scaffold package | `src/dl_pipeline/__init__.py`, `requirements_dl_pipeline.txt`, `tests/dl_pipeline/__init__.py` | `1b743a7d` |
| 1 | config.py | `src/dl_pipeline/config.py` (paths, hyperparams, locked seeds) | `02b29d45` |
| 2 | dataset index + split | `src/dl_pipeline/dataset.py` (`build_index`, `split_indices`), `tests/dl_pipeline/conftest.py`, `test_dataset.py`, `pytest.ini` | `5ae8ec74` |
| 3 | FrameDataset + face crop | append to `dataset.py` (`face_crop_from_landmarks`, `FrameDataset`) | `164db709` |
| 4 | WindowDataset | append to `dataset.py` (`build_windows`, `WindowDataset`) | `185da596` |
| 5 | build_cnn | `src/dl_pipeline/models.py` (`build_cnn`, `CNNFeatureExtractor`), `tests/dl_pipeline/test_models.py` | `39eb168a` |
| 6 | CNN-LSTM | append to `models.py` (`CNNLSTMHead`, `CNNLSTM`, `build_cnn_lstm`) | `58ba5c26` |
| 7 | metrics + plots | `src/dl_pipeline/train_eval.py` (`compute_metrics`, `save_confusion_plot`, `save_roc_plot`), `tests/dl_pipeline/test_train_eval.py` | `a772a6b0` |

**Test status:** `pytest tests/dl_pipeline/ -v` → 13 passed (4 dataset + 3 dataset + 3 dataset + 2 models + 2 models + 2 train_eval).

---

## What's missing — Tasks 8–13

Each task in the plan is fully self-contained: it lists the exact files to touch, the exact code to paste (tests AND implementation), the pytest commands to run, and the commit message. Just open the plan and follow the steps in order.

| # | Task | Plan section |
|---|---|---|
| 8 | `train_one_epoch`, `evaluate_cnn`, `train_cnn` (CNN training loop) | Task 8 |
| 9 | `cache_features` (per-video 576-dim feature cache) | Task 9 |
| 10 | `evaluate_lstm`, `train_lstm` (BiLSTM training on cached features) | Task 10 |
| 11 | `DLPredictor` class — `predict_frame`, `predict_window` (Member 5 fusion API) | Task 11 |
| 12 | `run_dl_pipeline.py` — CLI for `train_cnn / cache_features / train_lstm / evaluate` | Task 12 |
| 13 | End-to-end smoke run on real captured data (Colab/Kaggle) — produces `metrics.json`, ROC + confusion plots | Task 13 |

---

## How to resume (cold start)

```bash
# 1. Get on the branch
cd /Users/matthewmaingot/CV_Group_Project/Driver-Fatigue-Detection
git checkout member4/dl-pipeline
git pull

# 2. Sanity check — confirm what's done works
pip install -r src/dl_pipeline/requirements_dl_pipeline.txt
pytest tests/dl_pipeline/ -v        # expect 13 passed

# 3. Open the plan and start at Task 8
open docs/superpowers/plans/2026-04-21-dl-pipeline.md
```

For each remaining task (8 → 13), follow the plan's TDD recipe:

1. Append the failing tests from "Step 1" of the task into `tests/dl_pipeline/test_train_eval.py` (or the indicated test file).
2. Run `pytest …` from "Step 2" — confirm it fails with `ImportError` or similar.
3. Append the implementation from "Step 3" into the indicated source file.
4. Run `pytest …` from "Step 4" — confirm it passes.
5. Run the `git add … && git commit -m "…"` from "Step 5".

**Do not change the locked conventions** (split seed `42`, label map `{"trigger": 1, "non_trigger": 0}`, video regex, bbox logic — see the "Conventions" section at the top of the plan).

---

## Architecture — quick refresher

Two PyTorch models share a backbone:
- **CNN (Model A)** — MobileNetV3-Small fine-tuned on face crops (224×224). Per-frame binary classifier.
- **CNN-LSTM (Model B)** — CNN's frozen 576-dim features → BiLSTM (hidden=128) → 16-frame window (4 sec @ 4 fps) → many-to-one binary classifier.

Both evaluated on the **same test split** as Member 3's classical pipeline (`GroupShuffleSplit(test_size=0.25, random_state=42)`, grouped by video). This gives Member 5 (integration) a direct apples-to-apples comparison and a clean fusion contract via `DLPredictor.predict_frame()` / `predict_window()`.

---

## Compute notes

- Tasks 0–7 ran fine on CPU.
- Task 8's CNN training loop will benefit from GPU. For real-data training in Task 13, use **Google Colab** (free T4) or **Kaggle**: upload the repo, run `pip install -r src/dl_pipeline/requirements_dl_pipeline.txt`, then `python -m src.dl_pipeline.run_dl_pipeline train_cnn --epochs 20 --device auto --class-weights`.
- First run of `build_cnn()` downloads MobileNetV3-Small ImageNet weights (~10 MB).

---

## Hand-off to Member 5 (after Task 11)

Single import, mirrors classical SVM API:

```python
from src.dl_pipeline.predict import DLPredictor

pred = DLPredictor(
    cnn_ckpt="outputs/dl_pipeline/cnn_mobilenetv3.pt",
    lstm_ckpt="outputs/dl_pipeline/cnn_lstm.pt",   # optional
    device="cpu",
)
pred.predict_frame(frame_bgr, landmarks_px)         # → {"label", "proba", "model": "cnn"}
pred.predict_window(frames_bgr, landmarks_px_seq)   # → {"label", "proba", "model": "cnn_lstm"}
```

---

## Open the PR when ready

After Task 13:

```bash
gh pr create --title "Member 4 — Deep learning pipeline (CNN + CNN-LSTM)" \
  --body "Implements the DL branch per docs/superpowers/plans/2026-04-21-dl-pipeline.md. \
Test split locked to match Member 3 (GroupShuffleSplit, seed 42). \
Exposes DLPredictor for Member 5 fusion."
```
