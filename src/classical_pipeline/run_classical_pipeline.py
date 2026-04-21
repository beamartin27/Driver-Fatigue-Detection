"""
run_classical_pipeline.py
=========================
Member 3 — CLI entry point.

Runs the full classical pipeline end-to-end:
  1. Feature extraction (EAR + MAR + HOG) from landmarks + frames.
  2. Group-aware train/test split (by source video).
  3. Train + evaluate SVM, Random Forest, k-NN.
  4. Save models, confusion-matrix plots, and metrics.json.

Usage
-----
  # From the project root (after activating your venv):
  python -m src.classical_pipeline.run_classical_pipeline

  # With explicit paths:
  python -m src.classical_pipeline.run_classical_pipeline \\
      --frames data/frames --landmarks data/landmarks

  # Cache features to disk to skip re-extraction on the next run:
  python -m src.classical_pipeline.run_classical_pipeline --save-features

  # Reload cached features (fast re-training without re-extracting HOG):
  python -m src.classical_pipeline.run_classical_pipeline \\
      --load-features outputs/classical_pipeline/features.npz
"""

import argparse
import numpy as np
from pathlib import Path

from .dataset import build_dataset
from .train_eval import train_and_evaluate, DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Member 3 — Classical fatigue detection pipeline (EAR/MAR/HOG + SVM/RF/kNN)"
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="data/frames",
        help="Root folder containing trigger/ and non_trigger/ frame images (default: data/frames)",
    )
    parser.add_argument(
        "--landmarks",
        type=str,
        default="data/landmarks",
        help="Root folder containing trigger/ and non_trigger/ .npy landmark files (default: data/landmarks)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for models, confusion-matrix plots, and metrics.json",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of source videos to hold out for testing (default: 0.25)",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Cache extracted features to <output>/features.npz for faster re-runs",
    )
    parser.add_argument(
        "--load-features",
        type=str,
        default=None,
        metavar="PATH",
        help="Load a previously cached features.npz instead of re-extracting (skips HOG step)",
    )
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output)

    # ── Feature extraction ────────────────────────────────────────────────────
    if args.load_features:
        feat_path = Path(args.load_features)
        print(f"Loading cached features from {feat_path} ...")
        data   = np.load(str(feat_path), allow_pickle=True)
        X      = data["X"]
        y      = data["y"]
        groups = data["groups"]
        print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features per sample")
    else:
        print("Extracting features (EAR + MAR + HOG) ...")
        X, y, groups = build_dataset(
            frames_root    = Path(args.frames),
            landmarks_root = Path(args.landmarks),
            verbose        = True,
        )

        if args.save_features:
            out_dir.mkdir(parents=True, exist_ok=True)
            feat_path = out_dir / "features.npz"
            np.savez(str(feat_path), X=X, y=y, groups=groups)
            print(f"\nFeatures cached to: {feat_path}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    bad = ~np.isfinite(X)
    if bad.any():
        n_bad = int(bad.any(axis=1).sum())
        raise RuntimeError(
            f"{n_bad} samples contain NaN or Inf in their feature vectors. "
            "Check landmark extraction quality and ensure face ROIs are valid."
        )

    print(
        f"\nFeature matrix : {X.shape[0]} samples × {X.shape[1]} features"
        f"\nUnique videos  : {len(set(groups.tolist()))}"
        f"\ntrigger (1)    : {int((y == 1).sum())}"
        f"\nnon_trigger(0) : {int((y == 0).sum())}"
    )

    # ── Training & evaluation ─────────────────────────────────────────────────
    train_and_evaluate(
        X         = X,
        y         = y,
        groups    = groups,
        out_dir   = out_dir,
        test_size = args.test_size,
        verbose   = True,
    )


if __name__ == "__main__":
    main()
