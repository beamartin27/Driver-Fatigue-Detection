# Driver-Fatigue-Detection
Computer vision system for driver fatigue detection with a gesture-based activation mechanism.

---

## Member 1 — Gesture Activation

Recognizes an ordered hand-gesture sequence (OK then Peace) to switch the system from `inactive` to `activated`.

**Setup**
```bash
pip install -r src/gesture_activation/requirements_gesture_activation.txt
```

**Run (dataset evaluation)**
```bash
python -m src.gesture_activation.run_activation \
    --mode dataset \
    --data-root data/activation \
    --timeout-s 3.0
```

**Run (live webcam)**
```bash
python -m src.gesture_activation.run_activation \
    --mode webcam \
    --timeout-s 1.3 \
    --camera-id 0
```

See `src/gesture_activation/README.md` for full details.

---

## Member 2 — Face Detection & Landmark Extraction

Detects the driver's face and extracts 478 MediaPipe landmarks per frame.
Outputs `.npy` arrays consumed by the classical (Member 3) and DL (Member 4) pipelines.

**Setup (one-time)**
```bash
pip install mediapipe opencv-python
python landmark_extractor.py --download-model
```

**Extract landmarks from dataset frames**
```bash
python landmark_extractor.py \
    --frames data/frames \
    --output data/landmarks \
    --confidence 0.3
```

**Visualize landmarks on sample frames**
```bash
python visualize_landmarks.py --samples 5
```

---

## Member 3 — Classical Pipeline (EAR / MAR / HOG + SVM / RF / k-NN)

Computes per-frame features from Member 2 landmark outputs and trains three
classical classifiers for binary `trigger` vs `non_trigger` classification.

**Feature vector layout (1768 dims per frame)**

| Block       | Features                                    | Dims |
|-------------|---------------------------------------------|------|
| Scalars     | left_ear, right_ear, avg_ear, mar           | 4    |
| HOG         | 64x64 face ROI, 8x8 cells, 2x2 blocks, 9 bins | 1764 |

**Setup**
```bash
pip install -r src/classical_pipeline/requirements_classical_pipeline.txt
```

**Run full pipeline** (feature extraction + train + evaluate)
```bash
python -m src.classical_pipeline.run_classical_pipeline
```

**Cache features to skip HOG re-extraction on re-runs**
```bash
python -m src.classical_pipeline.run_classical_pipeline --save-features
```

**Reload cached features and retrain only**
```bash
python -m src.classical_pipeline.run_classical_pipeline \
    --load-features outputs/classical_pipeline/features.npz
```

**Expected outputs** (saved to `outputs/classical_pipeline/`)
```
metrics.json                 — accuracy / precision / recall / F1 per model
confusion_svm.png            — SVM confusion matrix
confusion_random_forest.png  — Random Forest confusion matrix
confusion_knn.png            — k-NN confusion matrix
svm.joblib                   — serialized SVM pipeline (scaler + clf)
random_forest.joblib         — serialized RF pipeline
knn.joblib                   — serialized k-NN pipeline
features.npz                 — cached feature matrix (if --save-features used)
```

**Baseline results on captured dataset (25% test split, group-by-video)**

| Model         | Accuracy | Precision | Recall | F1    |
|---------------|----------|-----------|--------|-------|
| SVM (RBF)     | 0.8634   | 0.9386    | 0.8449 | 0.8893|
| Random Forest | 0.8366   | 0.9142    | 0.8261 | 0.8679|
| k-NN (k=7)    | 0.7794   | 0.8707    | 0.7756 | 0.8204|

---

## Member 4 — Deep Learning (CNN + CNN-LSTM)

Fine-tuned MobileNetV3-Small plus an optional CNN–LSTM stack. Training entry points live in [`docs/superpowers/HANDOFF.md`](docs/superpowers/HANDOFF.md).

Inference for Member 5 consumes `src.dl_pipeline.predict.DLPredictor` (`predict_frame`, optional `predict_window`).

---

## Member 5 — Integration, Fusion & Real-Time Overlay

End-to-end stack: **capture → gesture gate → face landmarks → classical ∥ deep → fusion → alert HUD.**

### Prerequisites

```bash
pip install -r src/integration/requirements_integration.txt
python landmark_extractor.py --download-model
```

Expected checkpoints (adjust paths with CLI flags when needed):

- `outputs/classical_pipeline/svm.joblib` (`random_forest.joblib`, `knn.joblib`, or a custom `--classical-model-path` also work).
- For DL fusion: `outputs/dl_pipeline/cnn_mobilenetv3.pt`; optional temporal head `outputs/dl_pipeline/cnn_lstm.pt`.

### Run

```bash
python -m src.integration.run_system --camera-id 0 --timeout-s 4.5
```

| Flag | Effect |
|------|--------|
| `--video clips/demo.mp4` | File input instead of a webcam feed. |
| `--arm-on-start` | Bypasses the gesture handshake (automated QA / unattended capture rigs). |
| `--skip-dl` / `--skip-lstm` | Classical-only until CNN checkpoints exist / disable Bi-LSTM latency. |
| `--fusion-strategy {mean,max,min,weighted_mean,both}` | Late fusion rule (`weighted_mean` default with `--w-classical`, `--w-cnn`, `--w-lstm`). |
| `--fusion-threshold` | Alert when fused P(trigger) ≥ threshold. |
| `--reset-face-tracker-on-arm` | Immediately rebuild FaceLandmarker after gestures arm the system. |
| `--headless` | Suppress GUI windows (prints fused alerts instead). |

**Keys while the window has focus**

- **q** quit · **r** reset gesture sequence + trackers · **x** toggle gesture bypass

### Lightweight tests

```bash
pytest tests/integration/test_fusion.py -v
```

### Logistics / report reminders

Own the fused architecture narrative, reproducible run-books, fused-threshold rationale, residual failure cases (hands occluding face, low light, yawning-but-attentive, etc.), and the final edited demo once the team signs off on alert semantics.
