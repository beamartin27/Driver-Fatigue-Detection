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
