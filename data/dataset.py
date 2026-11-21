"""Loads the files written by data/collect.py, one whole episode per item.

Batches are padded to the longest episode in the batch. `mask` marks real frames.
Conventions (see envs/cue_recall.py): for an episode of L frames,
    obs[t]  : frame t (uint8, 64x64x3)
    act[t]  : action taken when seeing obs[t]; the last entry is -1
    rew[t]  : reward received on arriving at obs[t]; rew[0] = 0
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", gaps=None):
        p = Path(data_dir) / f"cue_recall_{split}"
        self.obs = np.load(f"{p}_obs.npy", mmap_mode="r")
        self.act = np.load(f"{p}_act.npy")
        self.rew = np.load(f"{p}_rew.npy")
        meta = np.load(f"{p}_meta.npz")
        self.meta = {k: meta[k] for k in meta.files}
        self.starts = self.meta["starts"]
        n = len(self.meta["gap"])
        self.index = np.arange(n) if gaps is None else np.where(np.isin(self.meta["gap"], gaps))[0]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e = int(self.index[i])
        s, t = int(self.starts[e]), int(self.starts[e + 1])
        m = self.meta
        return dict(
            obs=torch.from_numpy(np.array(self.obs[s:t])),
            act=torch.from_numpy(self.act[s:t].astype(np.int64)),
            rew=torch.from_numpy(self.rew[s:t].astype(np.float32)),
            gap=int(m["gap"][e]), cue=int(m["cue"][e]),
            reached=int(m["reached"][e]), success=int(m["success"][e]),
        )


def collate_episodes(items):
    B = len(items)
    T = max(len(it["act"]) for it in items)
    obs = torch.zeros(B, T, 64, 64, 3, dtype=torch.uint8)
    act = torch.zeros(B, T, dtype=torch.long)
    rew = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    length = torch.zeros(B, dtype=torch.long)
    for b, it in enumerate(items):
        L = len(it["act"])
        obs[b, :L], act[b, :L], rew[b, :L], mask[b, :L], length[b] = it["obs"], it["act"], it["rew"], True, L
    return dict(
        obs=obs, act=act, rew=rew, mask=mask, length=length,
        gap=torch.tensor([it["gap"] for it in items]),
        cue=torch.tensor([it["cue"] for it in items]),
        reached=torch.tensor([it["reached"] for it in items]),
        success=torch.tensor([it["success"] for it in items]),
    )
