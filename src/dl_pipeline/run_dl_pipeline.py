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
