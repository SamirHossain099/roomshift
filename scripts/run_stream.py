"""E1/E2: online adaptation on a held-out MMFi environment, streamed subject by subject, action by action.

    python scripts/run_stream.py --modality radar --hold E04 --seed 0 --label_every 0 50 200 \
        --methods source norm tent rigid supft

Protocol: batches of 16 consecutive frames; predict, score, then reveal labels on every K-th frame
(K = --label_every; 0 = none). Writes results/stream_<modality>_<hold>_s<seed>_K<K><tag>_<t>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.adapt import METHODS  # noqa: E402
from mmfi.data import AMP_MU, AMP_SD, ENVS, batches, build_tensors, concat, load_env  # noqa: E402
from mmfi.metrics import PoseMeter  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402


def _parse(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def run(method, T, bs, K, dev, iid=False, seed=0):
    order = np.lexsort((np.arange(len(T["pose"])), T["seq"].cpu().numpy()))  # by sequence, then time
    if iid:  # diagnostic control: random frame order (i.i.d. batches); not deployable
        order = np.random.default_rng(seed).permutation(order)
    meter, per_sub, lat, frame = PoseMeter(), {}, [], 0
    for b in batches(T, order, bs, False):
        n = len(b["pose"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pose, act = method.step(b)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000 / n)
        meter.update(pose, act, b["pose"], b["action"])
        for s in b["subject"].unique().tolist():
            m = b["subject"] == s
            per_sub.setdefault(s, PoseMeter()).update(pose[m], act[m], b["pose"][m], b["action"][m])
        lab = torch.zeros(n, dtype=torch.bool, device=dev)
        if K:
            lab = (torch.arange(frame, frame + n, device=dev) % K) == 0
        frame += n
        t0 = time.perf_counter()
        method.feedback(b, {"label": lab})
        torch.cuda.synchronize()
        lat[-1] += (time.perf_counter() - t0) * 1000 / n
    # median hides the gradient updates (they run on the few batches that carry a label); the mean amortises them
    return meter.result() | {"latency_ms_per_frame": float(np.median(lat)),
                             "latency_mean_ms_per_frame": float(np.mean(lat)), "latency_p99_ms_per_frame": float(np.percentile(lat, 99)),
                             "per_subject": {str(s): m.result() for s, m in per_sub.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["radar", "wifi"], default="radar")
    ap.add_argument("--hold", default="E04")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_every", nargs="+", type=int, default=[0])
    ap.add_argument("--methods", nargs="+", default=["source", "norm", "tent", "rigid", "supft"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--kw", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--iid", action="store_true", help="diagnostic: shuffled target order")
    a = ap.parse_args()
    dev = "cuda"
    csi = a.modality == "wifi"
    ck = torch.load(ROOT / "checkpoints" / f"{a.modality}_hold{a.hold}_s{a.seed}.pt", map_location=dev, weights_only=False)  # own files
    model = SenseNet(a.modality).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    T = build_tensors(load_env(a.hold), csi)
    src_stats = None
    if csi:
        src = concat([load_env(e) for e in ENVS if e != a.hold])
        src_stats = torch.as_tensor(((src["csi_amp"].astype(np.float32).mean((0, 2, 3)) - AMP_MU) / AMP_SD),
                                    device=dev, dtype=torch.float32)
    kw = {k: _parse(v) for k, v in (x.split("=", 1) for x in a.kw)}
    for K in a.label_every:
        res = {"modality": a.modality, "hold": a.hold, "seed": a.seed, "label_every": K, "bs": a.bs, "kw": a.kw, "iid": a.iid,
               "source_ckpt_cross": {k: float(v) for k, v in ck["cross"].items()}, "methods": {}}
        for name in a.methods:
            torch.manual_seed(a.seed)
            extra = {"src_stats": src_stats} if getattr(METHODS[name], "needs_source", False) else {}
            r = run(METHODS[name](model, **kw, **extra), T, a.bs, K, dev, iid=a.iid, seed=a.seed)
            res["methods"][name] = r
            print(f"[{a.modality} hold {a.hold} s{a.seed} K={K}] {name:16s} MPJPE {r['mpjpe_mm']:6.1f}  PA {r['pa_mpjpe_mm']:6.1f}  "
                  f"rel {r['rel_mpjpe_mm']:6.1f}  root {r['root_err_mm']:6.1f}  act {r['act_acc']:.3f}  {r['latency_ms_per_frame']:.2f} ms", flush=True)
        out = ROOT / "results" / f"stream_{a.modality}_{a.hold}_s{a.seed}_K{K}{a.tag}_{int(time.time())}.json"
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
