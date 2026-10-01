"""Pre-deployment check: does the frozen model locate the person? Measured on the training rooms only.

For each source model, the correlation between predicted and true root (joint centroid) over the frames of the
held-out subjects of its own training rooms (the within-room reference split, no target-room data), per axis, both
pooled and after removing each room's mean (pooled, a model that only recognizes the room scores well). If
this is high, the model localizes the person and an output correction should suffice after a room change; if it is
near zero, the model predicts a fixed position and needs target labels in its weights. Writes results/predeploy.json.

    python scripts/predeploy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.data import ENVS, batches, build_tensors, concat, load_env, split_subjects  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402


def main():
    D = {e: load_env(e) for e in ENVS}
    out = []
    for mod in ("radar", "wifi"):
        for hold in ENVS:
            src = concat([D[e] for e in ENVS if e != hold])
            _, va = split_subjects(src, 0.8, 0)          # the same split train_eval.py used
            T = build_tensors(src, mod == "wifi")
            for seed in (0, 1, 2):
                ck = torch.load(ROOT / "checkpoints" / f"{mod}_hold{hold}_s{seed}.pt", map_location="cuda", weights_only=False)
                model = SenseNet(mod).cuda()
                model.load_state_dict(ck["state"])
                model.eval()
                pr, gt, env = [], [], []
                with torch.no_grad():
                    for b in batches(T, va, 2048, False):
                        pr.append(model(b)[0].mean(1).cpu().numpy())
                        gt.append(b["pose"].mean(1).cpu().numpy())
                        env.append(b["env"].cpu().numpy())
                P, G, E = np.concatenate(pr), np.concatenate(gt), np.concatenate(env)
                corr = [float(np.corrcoef(P[:, j], G[:, j])[0, 1]) for j in range(3)]
                # pooled over rooms, a model that only recognizes the room scores well; remove each room's mean
                Pd, Gd = P.copy(), G.copy()
                for e in np.unique(E):
                    Pd[E == e] -= P[E == e].mean(0)
                    Gd[E == e] -= G[E == e].mean(0)
                dcorr = [float(np.corrcoef(Pd[:, j], Gd[:, j])[0, 1]) for j in range(3)]
                out.append({"modality": mod, "hold": hold, "seed": seed, "pooled_root_corr_xyz": corr,
                            "within_room_root_corr_xyz": dcorr})
                print(f"[{mod} hold {hold} s{seed}] corr(pred, true root) x/y/z pooled "
                      + " ".join(f"{c:+.2f}" for c in corr) + " | within each room "
                      + " ".join(f"{c:+.2f}" for c in dcorr), flush=True)
            del T
            torch.cuda.empty_cache()
    (ROOT / "results" / "predeploy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
