"""Memoryless control: a per-step MLP with NO state.

It sees only the current frame and action, never the past. It cannot remember the cue, so on the
cue probe / final-reward AUC it must sit at chance. If it does better, the task leaks the answer.
"""
import torch
from torch import nn

from models.cores.base import Core


class MemorylessCore(Core):
    def __init__(self, dim: int = 256, hidden: int = 512):
        super().__init__()
        self.dim, self.out_dim = dim, hidden
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, x, state=None):
        return self.net(x), None

    def step(self, x, state=None):
        return self.net(x), None
