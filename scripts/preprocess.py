"""Convert extracted MMFi (radar, Wi-Fi CSI, ground truth) into one compact npz per environment.

Per frame:
  radar   (P, 5) float16  up to P=64 mmWave points (x, y, z, doppler, intensity), zero padded
  n_pts   int16           number of real points
  csi_amp (3, 114, 10) float16 [dB], csi_pha (3, 114, 10) float16 [rad]; csi_ok marks frames with CSI
  pose    (17, 3) float32 3-D keypoints (ground truth)
  action (int, 0..26), subject (int, 1..40), seq (int, unique per E/S/A), frame (int)

    python scripts/preprocess.py            # all four environments, 8 worker processes
"""
import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RAW = Path(os.environ.get("MMFI_ROOT", r"N:\Datasets\MMFi\extracted\MMFi_Dataset"))
OUT = Path(__file__).resolve().parents[1] / "data" / "cache"
P = 64


def frame_no(name):
    return int(re.findall(r"\d+", name)[-1])


def one_sequence(args):
    import scipy.io as sio
    env, subj, act = args
    d = RAW / env / subj / act
    gt = np.load(d / "ground_truth.npy").astype(np.float32)          # (T, 17, 3)
    T = len(gt)
    rad = np.zeros((T, P, 5), np.float16)
    npts = np.zeros(T, np.int16)
    amp = np.zeros((T, 3, 114, 10), np.float16)
    pha = np.zeros((T, 3, 114, 10), np.float16)
    ok = np.zeros(T, bool)
    rfiles = {frame_no(f): f for f in os.listdir(d / "mmwave")} if (d / "mmwave").exists() else {}
    wfiles = {frame_no(f): f for f in os.listdir(d / "wifi-csi")} if (d / "wifi-csi").exists() else {}
    for t in range(T):
        fn = t + 1                                                    # frames are 1-indexed on disk
        if fn in rfiles:
            pts = np.fromfile(d / "mmwave" / rfiles[fn], dtype=np.float64)
            if pts.size % 5 == 0 and pts.size:
                pts = pts.reshape(-1, 5)
                if len(pts) > P:                                      # keep the strongest-intensity points
                    pts = pts[np.argsort(-pts[:, 4])[:P]]
                rad[t, : len(pts)] = pts
                npts[t] = len(pts)
        if fn in wfiles:
            m = sio.loadmat(d / "wifi-csi" / wfiles[fn])
            # amplitudes are in dB (0..~52), phases in rad; a few frames carry inf/NaN -> 0, then clip
            a = np.clip(np.nan_to_num(m["CSIamp"], nan=0.0, posinf=0.0, neginf=0.0), -100, 100)
            p = np.clip(np.nan_to_num(m["CSIphase"], nan=0.0, posinf=0.0, neginf=0.0), -np.pi, np.pi)
            if a.shape == (3, 114, 10):
                amp[t], pha[t], ok[t] = a, p, True
    return env, int(subj[1:]), int(act[1:]) - 1, gt, rad, npts, amp, pha, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["E01", "E02", "E03", "E04"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for env in a.envs:
        out = OUT / f"mmfi_{env}.npz"
        if out.exists():
            print("cached", out.name, flush=True)
            continue
        jobs = [(env, s, x) for s in sorted(os.listdir(RAW / env)) for x in sorted(os.listdir(RAW / env / s))]
        cols = {k: [] for k in ("pose", "radar", "n_pts", "csi_amp", "csi_pha", "csi_ok", "action", "subject", "seq", "frame")}
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (e, subj, act, gt, rad, npts, amp, pha, ok) in enumerate(ex.map(one_sequence, jobs, chunksize=2)):
                T = len(gt)
                seq_id = {"E01": 0, "E02": 1, "E03": 2, "E04": 3}[e] * 10000 + subj * 100 + act
                cols["pose"].append(gt), cols["radar"].append(rad), cols["n_pts"].append(npts)
                cols["csi_amp"].append(amp), cols["csi_pha"].append(pha), cols["csi_ok"].append(ok)
                cols["action"].append(np.full(T, act, np.int16)), cols["subject"].append(np.full(T, subj, np.int16))
                cols["seq"].append(np.full(T, seq_id, np.int32)), cols["frame"].append(np.arange(T, dtype=np.int16))
                if i % 50 == 0:
                    print(f"{env}: {i + 1}/{len(jobs)} sequences", flush=True)
        np.savez(out, **{k: np.concatenate(v) for k, v in cols.items()})
        print(f"wrote {out.name}: {sum(len(x) for x in cols['pose'])} frames from {len(jobs)} sequences", flush=True)


if __name__ == "__main__":
    main()
