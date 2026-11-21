"""Autoencoder sanity check for the shared encoder/decoder, on your collected data.

Trains encoder -> decoder to reproduce single frames (train split), then measures reconstruction
error on the val split, separately for the three kinds of frame:
    cue   : frame 0 (big colored block in the centre)
    gap   : blank frames with only the agent (a tiny 4x4 px square)
    goals : frames with the four colored corner goals (and the final border flash)
and compares each to the error of just predicting the empty background. A ratio well below 1
means the model really draws that kind of content. Also saves a side-by-side PNG.

Run from the project root:
    python -m models.ae_sanity
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.decoder import Decoder
from models.encoder import Encoder, to_float

TYPE_NAMES = ["cue (t=0)", "gap (agent only)", "goals visible"]
BG = 25 / 255.0


def frame_types(meta):
    starts, gap = meta["starts"], meta["gap"]
    types = np.zeros(int(starts[-1]), dtype=np.int8)
    for e in range(len(gap)):
        s = int(starts[e])
        types[s + 1 : s + 1 + int(gap[e])] = 1
        types[s + 1 + int(gap[e]) : int(starts[e + 1])] = 2
    return types


@torch.no_grad()
def evaluate(enc, dec, val_obs, types, dev, bs=256):
    enc.eval(); dec.eval()
    sums = np.zeros((3, 2)); counts = np.zeros(3)
    for s in range(0, len(val_obs), bs):
        x = torch.from_numpy(np.ascontiguousarray(val_obs[s : s + bs])).to(dev)
        target = to_float(x)
        err = ((dec(enc(x)) - target) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        base = ((BG - target) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        t = types[s : s + bs]
        for k in range(3):
            m = t == k
            sums[k] += [err[m].sum(), base[m].sum()]; counts[k] += m.sum()
    enc.train(); dec.train()
    return sums / np.maximum(counts, 1)[:, None]


@torch.no_grad()
def save_figure(enc, dec, val_obs, meta, dev, path):
    starts, gap = meta["starts"], meta["gap"]
    idx = []
    for e in range(4):  # cue frame, frame where goals appear, final frame
        idx += [int(starts[e]), int(starts[e]) + int(gap[e]) + 1, int(starts[e + 1]) - 1]
    x = torch.from_numpy(np.stack([val_obs[i] for i in idx])).to(dev)
    enc.eval(); dec.eval()
    rec = Decoder.to_uint8(dec(enc(x))).cpu().numpy()
    orig = x.cpu().numpy()
    top = np.concatenate(list(orig), axis=1)
    bottom = np.concatenate(list(rec), axis=1)
    img = Image.fromarray(np.concatenate([top, bottom], axis=0))
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"saved {path} (top row: original, bottom row: reconstruction)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/figures/ae_sanity.png")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_obs = np.load(f"{a.data}/cue_recall_train_obs.npy", mmap_mode="r")
    val_obs = np.load(f"{a.data}/cue_recall_val_obs.npy", mmap_mode="r")
    meta = np.load(f"{a.data}/cue_recall_val_meta.npz")
    types = frame_types(meta)

    enc, dec = Encoder().to(dev), Decoder().to(dev)
    n_enc = sum(p.numel() for p in enc.parameters()) / 1e6
    n_dec = sum(p.numel() for p in dec.parameters()) / 1e6
    print(f"device={dev} | encoder {n_enc:.2f}M params | decoder {n_dec:.2f}M params")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = np.random.default_rng(0)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = np.sort(rng.choice(len(train_obs), a.batch, replace=False))
        x = torch.from_numpy(train_obs[idx]).to(dev)
        loss = ((dec(enc(x)) - to_float(x)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == 1:
            print(f"step {step:>5} | loss {loss.item():.5f} | {time.time() - t0:.0f}s")

    res = evaluate(enc, dec, val_obs, types, dev)
    print("\nval reconstruction error (MSE per pixel) vs. predicting an empty background")
    print(f"{'frame type':<18}{'model':>10}{'background':>12}{'ratio':>8}")
    for k, name in enumerate(TYPE_NAMES):
        print(f"{name:<18}{res[k, 0]:>10.5f}{res[k, 1]:>12.5f}{res[k, 0] / res[k, 1]:>8.2f}")
    save_figure(enc, dec, val_obs, meta, dev, a.out)


if __name__ == "__main__":
    main()
