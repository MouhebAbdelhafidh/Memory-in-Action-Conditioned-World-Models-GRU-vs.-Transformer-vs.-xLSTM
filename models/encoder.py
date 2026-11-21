"""Shared frame encoder: (..., 64, 64, 3) frames -> (..., latent_dim) vectors.

Used unchanged by all three world-model variants (GRU / Transformer / xLSTM), so only the
dynamics core differs between experiments.

Input can be uint8 in [0, 255] (what data/collect.py writes) or float in [0, 1].
Any number of leading dims is allowed, e.g. (B, T, 64, 64, 3).
"""
import torch
from torch import nn


def to_float(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] -> float [0,1]; float input is assumed to be in [0,1] already."""
    if frames.dtype == torch.uint8:
        return frames.float() / 255.0
    return frames.float()


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 256, channels=(32, 64, 128, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        layers, c_in = [], 3
        for c in channels:  # each block halves the resolution: 64 -> 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(c_in, c, kernel_size=4, stride=2, padding=1),
                       nn.GroupNorm(8, c), nn.SiLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        side = 64 // (2 ** len(channels))
        self.fc = nn.Linear(c_in * side * side, latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        lead = frames.shape[:-3]
        x = to_float(frames.reshape(-1, *frames.shape[-3:]))  # (N, H, W, 3)
        x = x.permute(0, 3, 1, 2) - 0.5  # (N, 3, H, W), centred
        z = self.fc(self.conv(x).flatten(1))
        return z.reshape(*lead, self.latent_dim)