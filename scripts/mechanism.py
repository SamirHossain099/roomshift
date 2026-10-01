"""Why normalisation TTA destroys absolute pose: the normalisation statistics carry the person's position.

1. Probe: on held-out-room streams (batches of 16 consecutive frames, as deployed), regress the batch-mean
   ground-truth root (keypoint centroid) on the batch-mean input of each BatchNorm layer of the source model
   (ridge, 5-fold CV over sequences). High R^2 = the statistics a normalisation-TTA method re-estimates from the
   target encode where the person is, so re-centring them re-centres the person.
2. Behaviour: correlation over frames between predicted and true root (each axis), source vs norm-TTA.

    python scripts/mechanism.py --modality radar
Writes results/mechanism_<modality>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.adapt import METHODS, EMANorm  # noqa: E402
from mmfi.data import ENVS, batches, build_tensors, load_env  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402


def ridge_cv_r2(X, Y, groups, lam=1e-2, k=5):
    ug = np.unique(groups)
    fold = {g: i % k for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[g] for g in groups])
    pred = np.zeros_like(Y)
    for i in range(k):
        tr, te = f != i, f == i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd
        W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y[tr] - Y[tr].mean(0)))
        pred[te] = (X[te] - mu) / sd @ W + Y[tr].mean(0)
    return (1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "wifi"], default="radar")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = {}
    for hold in ENVS:
        ck = torch.load(ROOT / "checkpoints" / f"{a.modality}_hold{hold}_s{a.seed}.pt", map_location="cuda", weights_only=False)
        model = SenseNet(a.modality).cuda()
        model.load_state_dict(ck["state"])
        model.eval()
        T = build_tensors(load_env(hold), a.modality == "wifi")
        order = np.lexsort((np.arange(len(T["pose"])), T["seq"].cpu().numpy()))
        rec = {}
        for name in ("source", "norm"):
            meth = METHODS[name](model)
            norms = [m for m in meth.model.modules() if isinstance(m, EMANorm)]
            feats = {i: [] for i in range(len(norms))}
            hooks = [m.register_forward_pre_hook(lambda mod, inp, i=i: feats[i].append(
                inp[0].detach().float().mean([0] + list(range(2, inp[0].dim()))).cpu().numpy())) for i, m in enumerate(norms)]
            pr, gt, seq = [], [], []
            for b in batches(T, order, 16, False):
                pose, _ = meth.step(b)
                pr.append(pose.mean(1).cpu().numpy())
                gt.append(b["pose"].mean(1).cpu().numpy())
                seq.append(int(b["seq"][0]))
            for h in hooks:
                h.remove()
            P, G = np.concatenate(pr), np.concatenate(gt)
            corr = [float(np.corrcoef(P[:, j], G[:, j])[0, 1]) for j in range(3)]
            rec[name] = {"root_corr_xyz": corr, "pred_root_sd_mm": (1000 * P.std(0)).tolist(),
                         "true_root_sd_mm": (1000 * G.std(0)).tolist()}
            if name == "source":
                Gb = np.array([g.mean(0) for g in gt])
                rec["probe_r2_xyz_per_bn_layer"] = [ridge_cv_r2(np.array(feats[i]), Gb, np.array(seq)) for i in feats]
            print(f"[{a.modality} {hold}] {name:6s} corr(pred root, true root) x/y/z "
                  + " ".join(f"{c:+.2f}" for c in corr), flush=True)
        print(f"[{a.modality} {hold}] probe R^2 (x, y, z) per BN layer: "
              + "; ".join("/".join(f"{v:.2f}" for v in r) for r in rec["probe_r2_xyz_per_bn_layer"]), flush=True)
        out[hold] = rec
        del T
        torch.cuda.empty_cache()
    (ROOT / "results" / f"mechanism_{a.modality}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
