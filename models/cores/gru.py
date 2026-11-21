"""GRU core (RSSM-style recurrent baseline): one compact hidden vector, updated every step."""
import torch
from torch import nn

from models.cores.base import Core


class GRUCore(Core):
    def __init__(self, dim: int = 256, hidden: int = 512, layers: int = 1):
        super().__init__()
        self.dim, self.out_dim = dim, hidden
        self.gru = nn.GRU(dim, hidden, num_layers=layers, batch_first=True)

    def forward(self, x, state=None):
        h, state = self.gru(x, state)
        return h, state

    def step(self, x, state=None):
        h, state = self.gru(x.unsqueeze(1), state)
        return h.squeeze(1), state