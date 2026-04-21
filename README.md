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
