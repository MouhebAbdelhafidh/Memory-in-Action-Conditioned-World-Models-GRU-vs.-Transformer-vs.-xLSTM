"""Shared frame decoder: (..., in_dim) feature vectors -> (..., 64, 64, 3) frames in [0, 1].

`in_dim` is the size of whatever it decodes. In the world model it decodes the core's belief state
h_t (to predict the next frame); in the autoencoder sanity check it decodes the encoder's z_t.
Output is a plain linear head (not squashed), so values can slightly exceed [0, 1]; the training
loss is MSE against frames converted with encoder.to_float.
"""
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_dim: int = 256, channels=(256, 128, 64, 32)):
        super().__init__()
        self.in_dim = in_dim
        self.c0 = channels[0]
        self.side = 64 // (2 ** len(channels))  # 4
        self.fc = nn.Linear(in_dim, self.c0 * self.side * self.side)
        layers = []
        for i, c_in in enumerate(channels):  # each block doubles the resolution: 4 -> 8 -> 16 -> 32 -> 64
            c_out = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            layers += [nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                       nn.GroupNorm(8, c_out), nn.SiLU()]
        self.up = nn.Sequential(*layers)
        self.out = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        lead = feat.shape[:-1]
        x = self.fc(feat.reshape(-1, self.in_dim)).reshape(-1, self.c0, self.side, self.side)
        x = self.out(self.up(x)) + 0.5  # (N, 3, 64, 64), centred back to [0, 1]
        return x.permute(0, 2, 3, 1).reshape(*lead, 64, 64, 3)

    @staticmethod
    def to_uint8(frames: torch.Tensor) -> torch.Tensor:
        return (frames.clamp(0, 1) * 255).round().to(torch.uint8)