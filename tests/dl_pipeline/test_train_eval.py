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
