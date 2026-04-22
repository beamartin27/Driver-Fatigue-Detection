import torch
from src.dl_pipeline.models import build_cnn, CNNFeatureExtractor


def test_build_cnn_outputs_2_class_logits():
    model = build_cnn(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)


def test_cnn_feature_extractor_returns_576_dim():
    extractor = CNNFeatureExtractor(build_cnn(num_classes=2))
    x = torch.randn(2, 3, 224, 224)
    feats = extractor(x)
    assert feats.shape == (2, 576), f"Expected (2, 576), got {feats.shape}"


from src.dl_pipeline.models import build_cnn_lstm, CNNLSTMHead


def test_cnn_lstm_head_processes_window_to_logits():
    head = CNNLSTMHead(in_dim=576, hidden=128, num_classes=2)
    x = torch.randn(4, 16, 576)
    y = head(x)
    assert y.shape == (4, 2)


def test_build_cnn_lstm_processes_image_window_end_to_end():
    model = build_cnn_lstm(num_classes=2)
    x = torch.randn(2, 4, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)
