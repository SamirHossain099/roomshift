"""Per-subject root bias of the frozen source model in the held-out room: is the error a room constant?

For each held-out room and seed, the mean root error vector (predicted minus true joint centroid) of each of its ten
subjects. Decomposed into the room mean (what a correction fitted once per room can remove) and the spread between
subjects in the room (what it cannot). Also the slope of the bias on the subject's true mean position: a slope near
-1 means the model predicts nearly the same position for everyone (it does not localize the person); near 0 means
the bias is a constant offset. Writes results/bias_<modality>.json.

    python scripts/bias.py --modality radar
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.data import ENVS, batches, build_tensors, load_env  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "wifi"], default="radar")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    out = {"runs": []}
    for hold in ENVS:
        T = build_tensors(load_env(hold), a.modality == "wifi")
        sub = T["subject"].cpu().numpy()
        for seed in a.seeds:
            ck = torch.load(ROOT / "checkpoints" / f"{a.modality}_hold{hold}_s{seed}.pt", map_location="cuda", weights_only=False)
            model = SenseNet(a.modality).cuda()
            model.load_state_dict(ck["state"])
            model.eval()
            pr = []
            with torch.no_grad():
                for b in batches(T, np.arange(len(sub)), 2048, False):
                    pr.append(model(b)[0].mean(1).cpu().numpy())
            P, G = np.concatenate(pr), T["pose"].mean(1).cpu().numpy()
            subs = np.unique(sub)
            bias = np.array([(P - G)[sub == s].mean(0) for s in subs])          # per-subject mean error vector
            pos = np.array([G[sub == s].mean(0) for s in subs])
            room = bias.mean(0)
            slope = [float(np.polyfit(pos[:, j], bias[:, j], 1)[0]) for j in range(3)]
            out["runs"].append({"hold": hold, "seed": seed, "room_bias_mm": (1000 * room).tolist(),
                                "between_subject_sd_mm": (1000 * bias.std(0)).tolist(),
                                "subject_bias_mm": (1000 * bias).tolist(), "slope_bias_on_position": slope})
            print(f"[{a.modality} {hold} s{seed}] room bias {np.round(1000 * room)} mm | between-subject sd "
                  f"{np.round(1000 * bias.std(0))} mm | slope on position {np.round(slope, 2)}", flush=True)
        del T
        torch.cuda.empty_cache()
    r = out["runs"]
    out["summary"] = {"room_bias_norm_mm_mean": float(np.mean([np.linalg.norm(x["room_bias_mm"]) for x in r])),
                      "between_subject_sd_norm_mm_mean": float(np.mean([np.linalg.norm(x["between_subject_sd_mm"]) for x in r])),
                      "slope_mean_xyz": np.mean([x["slope_bias_on_position"] for x in r], 0).tolist()}
    print(out["summary"])
    (ROOT / "results" / f"bias_{a.modality}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
