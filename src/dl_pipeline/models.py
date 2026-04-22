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
