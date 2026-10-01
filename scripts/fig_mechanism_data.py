"""Per-frame predicted vs true root for the mechanism figure: radar frozen, radar under normalization TTA, Wi-Fi
frozen; held-out room E04, seed 0, the deployed stream order. Every 20th frame is kept. Writes
results/fig_mechanism.json (the figure reads only this file).

    python scripts/fig_mechanism_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.adapt import METHODS  # noqa: E402
from mmfi.data import batches, build_tensors, load_env  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402

HOLD, SEED, EVERY = "E04", 0, 20


def stream(modality, method):
    ck = torch.load(ROOT / "checkpoints" / f"{modality}_hold{HOLD}_s{SEED}.pt", map_location="cuda", weights_only=False)
    model = SenseNet(modality).cuda()
    model.load_state_dict(ck["state"])
    model.eval()
    T = build_tensors(load_env(HOLD), modality == "wifi")
    order = np.lexsort((np.arange(len(T["pose"])), T["seq"].cpu().numpy()))
    m = METHODS[method](model)
    pr, gt = [], []
    for b in batches(T, order, 16, False):
        pr.append(m.step(b)[0].mean(1).detach().cpu().numpy())
        gt.append(b["pose"].mean(1).cpu().numpy())
    P, G = np.concatenate(pr)[::EVERY], np.concatenate(gt)[::EVERY]
    return {"pred_m": P.round(4).tolist(), "true_m": G.round(4).tolist()}


def main():
    out = {"hold": HOLD, "seed": SEED, "every": EVERY, "axes": ["x lateral", "y vertical", "z depth"],
           "radar_source": stream("radar", "source"), "radar_norm": stream("radar", "norm"),
           "wifi_source": stream("wifi", "source")}
    (ROOT / "results" / "fig_mechanism.json").write_text(json.dumps(out))
    print("wrote results/fig_mechanism.json")


if __name__ == "__main__":
    main()
