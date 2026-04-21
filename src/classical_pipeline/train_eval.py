"""
train_eval.py
=============
Member 3 — Model training and evaluation.

Trains three classifiers on the feature dataset produced by dataset.py,
using a group-aware train/test split so that no source video contributes
frames to both sets (prevents temporal leakage).

Classifiers
-----------
- SVM (RBF kernel)
- RandomForestClassifier
- KNeighborsClassifier

All three are wrapped in a sklearn Pipeline that includes a StandardScaler
for uniformity (scaling is critical for SVM and k-NN; harmless for RF).

Artifacts saved to `out_dir` (default: outputs/classical_pipeline/)
---------------------------------------------------------------------------
  metrics.json                          — per-model accuracy/P/R/F1 + CM
  confusion_<model_name>.png            — confusion matrix heat-map
  <model_name>.joblib                   — serialised pipeline (scaler + clf)
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in headless environments
import matplotlib.pyplot as plt

import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DEFAULT_OUTPUT_DIR = Path("outputs/classical_pipeline")

# ── Model definitions ─────────────────────────────────────────────────────────


def _build_pipelines() -> dict[str, Pipeline]:
    """
    Return one sklearn Pipeline per model.

    Each pipeline:  StandardScaler  →  classifier
    Hyperparameters are reasonable baselines; tune with cross-val if needed.
    """
    return {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                n_jobs=-1,
            )),
        ]),
    }


# ── Train / test split ────────────────────────────────────────────────────────


def _group_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single group-aware train/test split using sklearn GroupShuffleSplit.

    Groups (source-video IDs) are kept intact — every frame from a given
    video goes entirely to train or entirely to test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx


# ── Evaluation helpers ────────────────────────────────────────────────────────


def _compute_metrics(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "model":            name,
        "n_test":           int(len(y_test)),
        "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":               round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
    }


def _save_confusion_matrix_plot(cm_list: list, model_name: str, out_dir: Path) -> None:
    cm = np.array(cm_list)
    class_labels = ["non_trigger (0)", "trigger (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    img = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(img, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
            )
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{model_name}.png", dpi=130)
    plt.close(fig)


# ── Public API ────────────────────────────────────────────────────────────────


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.25,
    verbose: bool = True,
) -> dict:
    """
    Run group-split → fit all models → evaluate → persist all artifacts.

    Parameters
    ----------
    X         : (N, F) float32 feature matrix (from dataset.build_dataset).
    y         : (N,)   int32   label vector.
    groups    : (N,)   object  source-video ID per sample.
    out_dir   : Directory where models, metrics, and plots are saved.
    test_size : Fraction of source videos held out for testing.
    verbose   : Print per-model results to stdout.

    Returns
    -------
    all_metrics : dict  {model_name: metrics_dict}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Split ────────────────────────────────────────────────────────────────
    train_idx, test_idx = _group_split(X, y, groups, test_size=test_size)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_groups = set(groups[train_idx])
    test_groups  = set(groups[test_idx])
    overlap      = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Group leakage: {len(overlap)} videos appear in both splits.")

    if verbose:
        n_pos_train = int((y_train == 1).sum())
        n_neg_train = int((y_train == 0).sum())
        n_pos_test  = int((y_test  == 1).sum())
        n_neg_test  = int((y_test  == 0).sum())
        print(
            f"\nTrain: {len(X_train)} samples ({n_pos_train} trigger / {n_neg_train} non_trigger)"
            f"  |  {len(train_groups)} videos"
        )
        print(
            f"Test:  {len(X_test)} samples ({n_pos_test} trigger / {n_neg_test} non_trigger)"
            f"  |  {len(test_groups)} videos"
        )

    # ── Train / evaluate ─────────────────────────────────────────────────────
    pipelines   = _build_pipelines()
    all_metrics = {}

    for name, pipeline in pipelines.items():
        if verbose:
            print(f"\n-- {name.upper()} --")
            print("  Training ...", end="", flush=True)

        pipeline.fit(X_train, y_train)

        if verbose:
            print(" done")

        metrics = _compute_metrics(pipeline, X_test, y_test, name)
        all_metrics[name] = metrics

        if verbose:
            print(f"  accuracy  : {metrics['accuracy']}")
            print(f"  precision : {metrics['precision']}")
            print(f"  recall    : {metrics['recall']}")
            print(f"  f1        : {metrics['f1']}")
            cm = metrics["confusion_matrix"]
            print(f"  confusion : {cm}")

        # Persist model
        joblib.dump(pipeline, out_dir / f"{name}.joblib")

        # Persist confusion matrix plot
        _save_confusion_matrix_plot(metrics["confusion_matrix"], name, out_dir)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    if verbose:
        print(f"\nAll artifacts saved to: {out_dir.resolve()}")
        print(f"  metrics.json")
        for name in pipelines:
            print(f"  {name}.joblib")
            print(f"  confusion_{name}.png")

    return all_metrics
