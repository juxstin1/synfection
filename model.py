"""GenoNet: log-mel spectrogram -> synth genome. The Genopatch network."""

import torch
import torch.nn as nn

from synth import N_PARAMS, N_MELS


def _block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
        nn.MaxPool2d(2),
    )


class GenoNet(nn.Module):
    def __init__(self, n_params=N_PARAMS):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_params),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)          # (B,1,n_mels,frames)
        return torch.sigmoid(self.head(self.features(x)))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
