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
