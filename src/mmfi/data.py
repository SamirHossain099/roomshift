"""MMFi environment arrays (scripts/preprocess.py) -> windows for training and streaming.

A sample is one frame t of one sequence with context:
  radar  (W*P, 6)  points of frames t-W+1..t (x, y, z, doppler, intensity, relative frame index), zero padded
  csi    (6, 114, 10) amplitude (dB, standardised) and phase (rad) of frame t
  pose   (17, 3)   ground-truth 3-D keypoints of frame t
  action int       0..26 (the whole sequence is one action)
  seq, env, subject, frame
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
ENVS = ("E01", "E02", "E03", "E04")
W = 5          # radar frames per window (MMFi radar is sparse: ~10-60 points per frame)
AMP_MU, AMP_SD = 40.0, 6.0


def load_env(env: str) -> dict:
    z = np.load(CACHE / f"mmfi_{env}.npz")
    d = {k: z[k] for k in z.files}
    d["env"] = np.full(len(d["pose"]), ENVS.index(env), np.int16)
    return d


def concat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


class Windows(torch.utils.data.Dataset):
    """Frame-level samples with a causal radar window; frames with t < W-1 reuse what exists."""

    def __init__(self, d: dict, idx=None, csi=True):
        self.d, self.csi = d, csi
        self.idx = np.arange(len(d["pose"])) if idx is None else np.asarray(idx)
        # start index of each frame's sequence, to stop windows at sequence boundaries
        seq = d["seq"]
        starts = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1]) + 1]
        first = np.zeros(len(seq), np.int64)
        for a, b in zip(starts, np.r_[starts[1:], len(seq)]):
            first[a:b] = a
        self.first = first

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        d = self.d
        lo = max(self.first[j], j - W + 1)
        pts = []
        for k, t in enumerate(range(lo, j + 1)):
            n = int(d["n_pts"][t])
            p = d["radar"][t, :n].astype(np.float32)
            pts.append(np.c_[p, np.full(n, (t - j) / W, np.float32)])
        pts = np.concatenate(pts) if pts else np.zeros((0, 6), np.float32)
        out = np.zeros((W * d["radar"].shape[1], 6), np.float32)
        out[: len(pts)] = pts[: len(out)]
        mask = np.zeros(len(out), np.float32)
        mask[: min(len(pts), len(out))] = 1
        s = {"radar": torch.from_numpy(out), "rmask": torch.from_numpy(mask),
             "pose": torch.from_numpy(d["pose"][j].astype(np.float32)),
             "action": torch.tensor(int(d["action"][j])), "seq": torch.tensor(int(d["seq"][j])),
             "env": torch.tensor(int(d["env"][j])), "subject": torch.tensor(int(d["subject"][j]))}
        if self.csi:
            amp = (d["csi_amp"][j].astype(np.float32) - AMP_MU) / AMP_SD
            pha = d["csi_pha"][j].astype(np.float32) / np.pi
            s["csi"] = torch.from_numpy(np.concatenate([amp, pha], 0))
        return s


def split_subjects(d: dict, frac=0.8, seed=0):
    """Within-environment reference split by subject (no subject in both halves)."""
    subs = np.unique(d["subject"])
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    tr = np.isin(d["subject"], subs[: int(len(subs) * frac)])
    return np.flatnonzero(tr), np.flatnonzero(~tr)


def build_tensors(d: dict, csi: bool, dev="cuda") -> dict:
    """Vectorised, GPU-resident version of Windows: every frame's radar window, CSI, labels as tensors.

    Radar window: frames t, t-1, ..., t-W+1 clipped at the sequence start; clipped (duplicate) frames are
    masked out. Memory: N x (W*P) x 6 float16 for radar (~0.9 GB for 240k frames), N x 6 x 114 x 10
    float16 for CSI (~3.3 GB)."""
    n, P = len(d["pose"]), d["radar"].shape[1]
    seq = d["seq"]
    starts = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1]) + 1]
    first = np.repeat(starts, np.diff(np.r_[starts, n]))
    t = np.arange(n)
    rad = np.zeros((n, W * P, 6), np.float16)
    mask = np.zeros((n, W * P), np.float16)
    for w in range(W):
        src = t - w
        valid = src >= first
        src = np.maximum(src, first)
        rad[:, w * P:(w + 1) * P, :5] = d["radar"][src]
        rad[:, w * P:(w + 1) * P, 5] = -w / W
        m = (np.arange(P)[None, :] < d["n_pts"][src][:, None]) & valid[:, None]
        mask[:, w * P:(w + 1) * P] = m
    out = {"radar": torch.as_tensor(rad, device=dev), "rmask": torch.as_tensor(mask, device=dev),
           "pose": torch.as_tensor(d["pose"], device=dev), "action": torch.as_tensor(d["action"].astype(np.int64), device=dev),
           "seq": torch.as_tensor(d["seq"].astype(np.int64), device=dev), "env": torch.as_tensor(d["env"].astype(np.int64), device=dev),
           "subject": torch.as_tensor(d["subject"].astype(np.int64), device=dev)}
    if csi:
        amp = (d["csi_amp"].astype(np.float16) - np.float16(AMP_MU)) / np.float16(AMP_SD)
        pha = d["csi_pha"].astype(np.float16) / np.float16(np.pi)
        out["csi"] = torch.as_tensor(np.concatenate([amp, pha], 1), device=dev)
    return out


def batches(T: dict, idx, bs: int, shuffle: bool, seed: int = 0):
    """Yield float32 batches from GPU tensors (index subset `idx`)."""
    idx = torch.as_tensor(np.asarray(idx), device=T["pose"].device)
    if shuffle:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
    for i in range(0, len(idx) - (bs if shuffle else 0) + (0 if shuffle else bs), bs):
        j = idx[i:i + bs]
        if len(j) == 0:
            break
        yield {k: (v[j].float() if v.dtype == torch.float16 else v[j]) for k, v in T.items()}
