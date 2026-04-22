"""PyTorch model definitions: per-frame CNN and CNN-LSTM."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from . import config


def build_cnn(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    """ImageNet-pretrained MobileNetV3-Small with a fresh `num_classes` head."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features  # 1024
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class CNNFeatureExtractor(nn.Module):
    """Extracts the 576-dim pooled feature vector from MobileNetV3-Small."""

    def __init__(self, cnn: nn.Module):
        super().__init__()
        self.features = cnn.features
        self.avgpool = cnn.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 576)


class CNNLSTMHead(nn.Module):
    """BiLSTM over a sequence of CNN features -> many-to-one classification.

    Hidden=128 -> 256 after concat -> 2-class head.
    """

    def __init__(self, in_dim: int = 576, hidden: int = config.LSTM_HIDDEN,
                 dropout: float = config.LSTM_DROPOUT, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # (B, T, 2H)
        last = out[:, -1, :]          # many-to-one
        return self.classifier(self.dropout(last))


class CNNLSTM(nn.Module):
    """End-to-end CNN-LSTM for inference: applies CNN per frame, then BiLSTM."""

    def __init__(self, extractor: CNNFeatureExtractor, head: CNNLSTMHead):
        super().__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.extractor(x).view(b, t, -1)
        return self.head(feats)


def build_cnn_lstm(num_classes: int = config.NUM_CLASSES, cnn: nn.Module | None = None) -> nn.Module:
    """Build a fresh CNN+LSTM. Pass an existing `cnn` to reuse fine-tuned weights."""
    if cnn is None:
        cnn = build_cnn(num_classes=num_classes)
    extractor = CNNFeatureExtractor(cnn)
    head = CNNLSTMHead(in_dim=576, num_classes=num_classes)
    return CNNLSTM(extractor, head)
