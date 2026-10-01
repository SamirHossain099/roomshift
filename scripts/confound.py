"""Room vs subject: is the cross-room root shift a property of the room or of who stands in it? (model-free)

MM-Fi gives each room its own 10 subjects, so a room effect and a subject effect are confounded. For the
ground-truth root (keypoint centroid), decompose the spread of per-subject mean positions into a between-room
part (room means) and a within-room, between-subject part, per axis. Writes results/confound.json.

    python scripts/confound.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.data import ENVS, load_env  # noqa: E402

AXES = ("x (lateral)", "y (vertical)", "z (depth)")


def main():
    subj, rad = {}, {}   # env -> subject -> mean GT root (3,) / mean radar centroid (3,)
    for e in ENVS:
        d = load_env(e)
        root = d["pose"].mean(1)
        n = d["n_pts"].astype(np.float64)
        ok = n > 0
        cen = d["radar"][:, :, :3].astype(np.float64).sum(1)[ok] / n[ok, None]
        subj[e] = {int(s): root[d["subject"] == s].mean(0) for s in np.unique(d["subject"])}
        rad[e] = {int(s): cen[d["subject"][ok] == s].mean(0) for s in np.unique(d["subject"])}
    room_mean = {e: np.mean(list(v.values()), 0) for e, v in subj.items()}
    grand = np.mean(list(room_mean.values()), 0)
    between = np.mean([(room_mean[e] - grand) ** 2 for e in ENVS], 0)                       # var of room means
    within = np.mean([np.var(list(v.values()), 0) for v in subj.values()], 0)               # mean within-room var
    out = {"room_mean_m": {e: room_mean[e].round(4).tolist() for e in ENVS},
           "subject_mean_m": {e: {s: p.round(4).tolist() for s, p in v.items()} for e, v in subj.items()},
           "between_room_sd_mm": (1000 * np.sqrt(between)).round(1).tolist(),
           "within_room_between_subject_sd_mm": (1000 * np.sqrt(within)).round(1).tolist(),
           "room_share_of_variance": (between / (between + within)).round(3).tolist()}
    for i, a in enumerate(AXES):
        print(f"{a:13s} between-room sd {out['between_room_sd_mm'][i]:6.1f} mm   within-room between-subject sd "
              f"{out['within_room_between_subject_sd_mm'][i]:6.1f} mm   room share {out['room_share_of_variance'][i]:.2f}")
    for e in ENVS:
        print(e, "room mean", out["room_mean_m"][e])
    out["placement"] = placement_test(rad, subj)
    (ROOT / "results" / "confound.json").write_text(json.dumps(out, indent=1))


def placement_test(rad, subj, n_perm=10000, seed=0):
    """Sensor placement vs where people stand. Per subject: mean radar point-cloud centroid c_s (radar frame)
    and mean ground-truth root r_s (pose frame). Fit r_s = A c_s + b over all 40 subjects (one shared affine
    radar-to-pose map) and r_s = A c_s + b_room (a per-room offset). If the radar-to-pose transform is the same
    in every room, the room offsets add nothing beyond chance; the permutation null shuffles room labels across
    subjects. The subject's own position is in c_s, so a room offset here is not 'where people stand'."""
    C, Rt, room = [], [], []
    for k, e in enumerate(ENVS):
        for s in subj[e]:
            C.append(rad[e][s])
            Rt.append(subj[e][s])
            room.append(k)
    C, Rt, room = np.array(C), np.array(Rt), np.array(room)

    def sse(lab):
        X = np.c_[C, np.eye(len(ENVS))[lab]]          # shared A, per-group intercept
        W, *_ = np.linalg.lstsq(X, Rt, rcond=None)
        return ((X @ W - Rt) ** 2).sum(0), W

    X0 = np.c_[C, np.ones(len(C))]
    W0, *_ = np.linalg.lstsq(X0, Rt, rcond=None)
    sse0 = ((X0 @ W0 - Rt) ** 2).sum(0)
    sse1, W1 = sse(room)
    gain = (sse0 - sse1).sum()
    rng = np.random.default_rng(seed)
    null = np.array([(sse0 - sse(rng.permutation(room))[0]).sum() for _ in range(n_perm)])
    off = W1[3:] - W1[3:].mean(0)                       # per-room offsets relative to their mean
    res = {"n_subjects": len(C),
           "resid_sd_shared_mm": (1000 * np.sqrt(sse0 / len(C))).round(1).tolist(),
           "resid_sd_room_offset_mm": (1000 * np.sqrt(sse1 / len(C))).round(1).tolist(),
           "room_offset_mm": {e: (1000 * off[k]).round(1).tolist() for k, e in enumerate(ENVS)},
           "perm_p": float((1 + (null >= gain).sum()) / (1 + n_perm))}
    print("\nplacement test (per-subject radar centroid -> GT root, shared affine map +/- per-room offset)")
    print("  residual sd shared map (x, y, z) mm:", res["resid_sd_shared_mm"])
    print("  residual sd with room offsets   mm:", res["resid_sd_room_offset_mm"])
    print("  room offsets mm:", res["room_offset_mm"])
    print(f"  permutation p (room labels shuffled across subjects, {n_perm}): {res['perm_p']:.4f}")
    return res


if __name__ == "__main__":
    main()
