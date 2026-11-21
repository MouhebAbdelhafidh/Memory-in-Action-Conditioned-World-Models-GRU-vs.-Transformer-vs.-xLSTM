"""Interface every dynamics core must implement (GRU, Transformer, xLSTM, memoryless control).

Only the core changes between experiments; encoder, decoder, heads, losses and data are shared.

    dim      : size of the input vectors the core receives (set by the world model)
    out_dim  : size of the belief state h_t the core returns (the decoder and the probes read this)

    init_state(batch_size, device) -> state     # None means "start fresh"
    forward(x_seq, state=None)     -> (h_seq, state)   # x_seq: (B, T, dim) -> h_seq: (B, T, out_dim); used in training
    step(x_t, state=None)          -> (h_t, state)     # x_t: (B, dim) -> h_t: (B, out_dim); used in planning / efficiency tests

`forward` over a sequence and `step` called T times must give the same h (checked in the tests).
"""
from torch import nn


class Core(nn.Module):
    out_dim: int

    def init_state(self, batch_size: int, device=None):
        return None

    def forward(self, x, state=None):
        raise NotImplementedError

    def step(self, x, state=None):
        raise NotImplementedError
