"""E3b: tune the supervised online fine-tuning baseline on all four held-out rooms (seed 0, 1% labels).

Grid: lr x gradient steps per labelled batch x which weights (all / heads only). The grid is scored on the
test streams themselves, i.e. oracle tuning that favours the baseline; the calibration methods get no such
tuning (window 64, alpha 0.3, ridge 1e-3 fixed before the final runs).

    python scripts/sweep_supft.py --modality radar
Writes results/sweep_supft_<modality>_s<seed>_K<K>.json (one record per config and room).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from mmfi.adapt import SupFT  # noqa: E402
from mmfi.data import ENVS, build_tensors, load_env  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402
from run_stream import run  # noqa: E402

# (part, lr, steps). Heads-only fine-tuning was tried on E01 only (lr 1e-4 to 1e-2, 1 and 4 steps; best radar 136.5,
# Wi-Fi 234.2 mm, both worse than all weights) and dropped from the other rooms.
GRID = {"radar": [("all", lr, s) for lr in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4) for s in (1, 4)]
                 + [("all", 1e-6, 16), ("all", 3e-6, 16), ("all", 3e-7, 64)],
        "wifi": [("all", lr, s) for lr in (1e-4, 3e-4, 1e-3, 3e-3) for s in (1, 4)]
                + [("all", 3e-4, 16), ("all", 1e-3, 16)]
                + [("all", 3e-3, 16), ("all", 1e-3, 64), ("all", 3e-4, 64)]
                + [("all", 3e-3, 64), ("all", 1e-3, 256), ("all", 3e-4, 256)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "wifi"], default="radar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_every", type=int, default=100)
    a = ap.parse_args()
    out = ROOT / "results" / f"sweep_supft_{a.modality}_s{a.seed}_K{a.label_every}.json"
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["hold"], r["part"], r["lr"], r["steps"]) for r in recs}
    for hold in ENVS:
        ck = torch.load(ROOT / "checkpoints" / f"{a.modality}_hold{hold}_s{a.seed}.pt", map_location="cuda", weights_only=False)
        model = SenseNet(a.modality).cuda()
        model.load_state_dict(ck["state"])
        model.eval()
        T = build_tensors(load_env(hold), a.modality == "wifi")
        for part, lr, steps in GRID[a.modality]:
            if (hold, part, lr, steps) in done:
                continue
            torch.manual_seed(a.seed)
            r = run(SupFT(model, lr=lr, steps=steps, part=part), T, 16, a.label_every, "cuda")
            r.pop("per_subject")
            recs.append({"hold": hold, "part": part, "lr": lr, "steps": steps, **r})
            print(f"[{a.modality} {hold}] {part:4s} lr {lr:g} steps {steps}: MPJPE {r['mpjpe_mm']:.1f}", flush=True)
            out.write_text(json.dumps(recs, indent=1))
        del T
        torch.cuda.empty_cache()
    print("\nmean over rooms:")
    keys = sorted({(r["part"], r["lr"], r["steps"]) for r in recs})
    for k in keys:
        v = [r["mpjpe_mm"] for r in recs if (r["part"], r["lr"], r["steps"]) == k]
        if len(v) == len(ENVS):
            print(f"  {k[0]:4s} lr {k[1]:g} steps {k[2]}: {np.mean(v):.1f}")


if __name__ == "__main__":
    main()
