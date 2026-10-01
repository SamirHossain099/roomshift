"""E0: leave-one-environment-out (LOEO) training and source-only evaluation, radar or Wi-Fi.

For each held-out environment E: train on the other three (all subjects), evaluate on E; also a
within-environment reference (train/test split by subject inside the same three environments).
Saves the model per held-out environment for the streaming experiments.

    python scripts/train_eval.py --modality radar --epochs 20 --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.data import ENVS, batches, build_tensors, concat, load_env, split_subjects  # noqa: E402
from mmfi.metrics import PoseMeter  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402


def train(model, T, idx, epochs, lr, bs, w_act=0.1, seed=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for b in batches(T, idx, bs, True, seed * 1000 + ep):
            pose, act = model(b)
            loss = F.l1_loss(pose, b["pose"]) + w_act * F.cross_entropy(act, b["action"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def evaluate(model, T, idx=None):
    m = PoseMeter()
    idx = np.arange(len(T["pose"])) if idx is None else idx
    for b in batches(T, idx, 2048, False):
        pose, act = model(b)
        m.update(pose, act, b["pose"], b["action"])
    return m.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "wifi"], default="radar")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--hold", nargs="+", default=list(ENVS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda"
    csi = a.modality == "wifi"
    D = {e: load_env(e) for e in ENVS}
    res = {"modality": a.modality, "epochs": a.epochs, "runs": []}
    for hold in a.hold:
        src = concat([D[e] for e in ENVS if e != hold])
        tr, va = split_subjects(src, 0.8, 0)
        T_src = build_tensors(src, csi)
        T_tgt = build_tensors(D[hold], csi)
        for seed in a.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t = time.time()
            model = SenseNet(a.modality).to(dev)
            train(model, T_src, tr, a.epochs, a.lr, a.bs, seed=seed)
            within = evaluate(model, T_src, va)
            cross = evaluate(model, T_tgt)
            ck = ROOT / "checkpoints" / f"{a.modality}_hold{hold}_s{seed}.pt"
            ck.parent.mkdir(exist_ok=True)
            torch.save({"state": model.state_dict(), "modality": a.modality, "hold": hold, "seed": seed,
                        "within": within, "cross": cross}, ck)
            res["runs"].append({"hold": hold, "seed": seed, "within": within, "cross": cross})
            print(f"[{a.modality}] hold {hold} seed {seed} ({time.time() - t:.0f}s)  "
                  f"within-env(subjects held out) MPJPE {within['mpjpe_mm']:.0f} PA {within['pa_mpjpe_mm']:.0f} "
                  f"rel {within['rel_mpjpe_mm']:.0f} root {within['root_err_mm']:.0f} act {within['act_acc']:.3f} | "
                  f"cross-env MPJPE {cross['mpjpe_mm']:.0f} PA {cross['pa_mpjpe_mm']:.0f} rel {cross['rel_mpjpe_mm']:.0f} "
                  f"root {cross['root_err_mm']:.0f} act {cross['act_acc']:.3f}", flush=True)
        del T_src, T_tgt
        torch.cuda.empty_cache()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"E0_{a.modality}{a.tag}_{int(time.time())}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
