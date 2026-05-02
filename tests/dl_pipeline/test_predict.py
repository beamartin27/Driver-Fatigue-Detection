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
