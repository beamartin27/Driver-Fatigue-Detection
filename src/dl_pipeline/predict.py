"""Inference API consumed by Member 5's fusion layer.

Mirrors the shape of the classical SVM's predict / predict_proba so the
fusion layer can treat the DL branch identically to the classical branch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torchvision import transforms

from . import config
from .dataset import face_crop_from_landmarks
from .models import build_cnn, CNNFeatureExtractor, CNNLSTMHead


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _eval_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class DLPredictor:
    """Loads the trained CNN (and optional LSTM head) for inference."""

    def __init__(
        self,
        cnn_ckpt: Path,
        lstm_ckpt: Path | None = None,
        lstm_hidden: int = config.LSTM_HIDDEN,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.cnn = build_cnn(num_classes=config.NUM_CLASSES)
        self.cnn.load_state_dict(torch.load(str(cnn_ckpt), map_location=self.device))
        self.cnn.train(False)
        self.cnn.to(self.device)
        self.extractor = CNNFeatureExtractor(self.cnn)
        self.extractor.train(False)
        self.extractor.to(self.device)
        self.lstm: CNNLSTMHead | None = None
        if lstm_ckpt is not None:
            self.lstm = CNNLSTMHead(in_dim=576, hidden=lstm_hidden, num_classes=config.NUM_CLASSES)
            self.lstm.load_state_dict(torch.load(str(lstm_ckpt), map_location=self.device))
            self.lstm.train(False)
            self.lstm.to(self.device)
        self.transform = _eval_transform()

    def _to_tensor(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> torch.Tensor:
        crop = face_crop_from_landmarks(frame_bgr, landmarks_px)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> dict:
        x = self._to_tensor(frame_bgr, landmarks_px).unsqueeze(0).to(self.device)
        logits = self.cnn(x)
        proba = torch.softmax(logits, dim=1)[0, 1].item()
        label = int(logits.argmax(dim=1).item())
        return {"label": label, "proba": float(proba), "model": "cnn"}

    @torch.no_grad()
    def predict_window(
        self,
        frames_bgr: Sequence[np.ndarray],
        landmarks_px_seq: Sequence[np.ndarray],
    ) -> dict:
        if self.lstm is None:
            raise RuntimeError("DLPredictor was constructed without an LSTM checkpoint; "
                               "pass lstm_ckpt= to use predict_window().")
        if len(frames_bgr) != len(landmarks_px_seq):
            raise ValueError("frames_bgr and landmarks_px_seq must be the same length")
        tensors = [self._to_tensor(f, lm) for f, lm in zip(frames_bgr, landmarks_px_seq)]
        x = torch.stack(tensors).to(self.device)
        feats = self.extractor(x).unsqueeze(0)        # (1, T, 576)
        logits = self.lstm(feats)
        proba = torch.softmax(logits, dim=1)[0, 1].item()
        label = int(logits.argmax(dim=1).item())
        return {"label": label, "proba": float(proba), "model": "cnn_lstm"}
