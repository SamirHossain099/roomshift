"""Model-free look at what differs between MM-Fi environments.

1. Ground-truth pose: per-environment mean root (keypoint centroid) position and spread. Is the
   subject placed differently relative to the pose coordinate frame in each room?
2. Radar: per-environment relation between the radar point-cloud centroid and the ground-truth root.
   If the radar-to-pose-frame transform differs per room, a fitted affine map from radar centroid to
   root will transfer poorly across rooms but well within a room.
3. Wi-Fi: per-environment CSI amplitude mean / spread per antenna (static multipath signature).

    python scripts/sanity.py
Writes results/sanity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.data import ENVS, load_env  # noqa: E402


def radar_centroid(d):
    n = d["n_pts"].astype(float)
    s = (d["radar"][:, :, :3].astype(np.float64)).sum(1)
    c = np.full((len(n), 3), np.nan)
    ok = n > 0
    c[ok] = s[ok] / n[ok, None]
    return c


def main():
    out = {}
    D = {e: load_env(e) for e in ENVS if (ROOT / "data" / "cache" / f"mmfi_{e}.npz").exists()}
    print("environments available:", list(D))
    print("\n1. ground-truth root (keypoint centroid), metres: mean (x, y, z) and std")
    for e, d in D.items():
        r = d["pose"].mean(1)
        out.setdefault(e, {})["gt_root_mean"] = r.mean(0).tolist()
        out[e]["gt_root_std"] = r.std(0).tolist()
        print(f"  {e}: mean {np.round(r.mean(0), 3)}  std {np.round(r.std(0), 3)}  frames {len(r)}")

    print("\n2. radar centroid -> GT root: per-env affine fit, within-env and cross-env error (mm)")
    C = {e: radar_centroid(d) for e, d in D.items()}
    fits = {}
    for e, d in D.items():
        c, r = C[e], d["pose"].mean(1)
        ok = np.isfinite(c).all(1)
        X = np.c_[c[ok], np.ones(ok.sum())]
        A, *_ = np.linalg.lstsq(X, r[ok], rcond=None)
        fits[e] = A
        print(f"  {e}: radar points/frame {d['n_pts'].mean():.1f}; radar centroid mean {np.round(np.nanmean(c, 0), 3)}")
    names = list(D)
    print("       " + "".join(f"{b:>9s}" for b in names))
    for a in names:
        row = []
        for b in names:
            c, r = C[b], D[b]["pose"].mean(1)
            ok = np.isfinite(c).all(1)
            pred = np.c_[c[ok], np.ones(ok.sum())] @ fits[a]
            row.append(1000 * np.linalg.norm(pred - r[ok], axis=1).mean())
        out[a]["radar_root_affine_err_mm"] = dict(zip(names, row))
        print(f"  {a}: " + "".join(f"{v:9.0f}" for v in row))

    print("\n3. Wi-Fi CSI amplitude (dB) per antenna: mean / std over frames with CSI")
    for e, d in D.items():
        ok = d["csi_ok"]
        a = d["csi_amp"][ok].astype(np.float32)
        m = a.mean((0, 2, 3))
        s = a.std((0, 2, 3))
        out[e]["csi_amp_mean"] = m.tolist()
        print(f"  {e}: CSI frames {ok.mean():.2f}; mean {np.round(m, 2)}  std {np.round(s, 2)}")
    (ROOT / "results" / "sanity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
