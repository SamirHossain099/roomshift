"""E4: compute cost per frame, amortised over the stream (predict + feedback, gradient updates included), on GPU and
on one CPU thread. Held-out room E04, seed 0, the first N frames of the stream, batches of 16, one label every 100
frames; supervised fine-tuning at the tuned settings. Writes results/latency.json.

The GPU is shared with other jobs on this machine, so GPU times are upper bounds; the CPU run is single-threaded.

    python scripts/latency.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfi.adapt import METHODS  # noqa: E402
from mmfi.data import batches, build_tensors, load_env  # noqa: E402
from mmfi.models import SenseNet  # noqa: E402

HOLD, SEED, K, N = "E04", 0, 100, 4000
TUNED = {"radar": {"lr": 3e-6, "steps": 4}, "wifi": {"lr": 1e-3, "steps": 64}}
METHODS_TIMED = ["source", "smooth", "shift", "rigid", "affine", "affine+smooth", "supft"]


def timed(method, T, dev):
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    order = np.lexsort((np.arange(len(T["pose"])), T["seq"].cpu().numpy()))[:N]
    frame, total, upd = 0, 0.0, []
    for b in batches(T, order, 16, False):
        n = len(b["pose"])
        lab = (torch.arange(frame, frame + n, device=dev) % K) == 0
        frame += n
        sync()
        t0 = time.perf_counter()
        method.step(b)
        sync()
        t1 = time.perf_counter()
        method.feedback(b, {"label": lab})
        sync()
        t2 = time.perf_counter()
        total += t2 - t0
        if lab.any():
            upd.append(t2 - t1)
    return {"ms_per_frame_amortised": 1000 * total / frame, "ms_per_update": 1000 * float(np.median(upd)),
            "frames": frame}


def main():
    out = {}
    for dev in ("cuda", "cpu"):
        if dev == "cpu":
            torch.set_num_threads(1)
        for mod in ("radar", "wifi"):
            ck = torch.load(ROOT / "checkpoints" / f"{mod}_hold{HOLD}_s{SEED}.pt", map_location=dev, weights_only=False)
            model = SenseNet(mod).to(dev)
            model.load_state_dict(ck["state"])
            model.eval()
            T = build_tensors(load_env(HOLD), mod == "wifi", dev=dev)
            T = {k: v[: N + 2000] for k, v in T.items()}   # the stream's first frames come first in sequence order
            for name in METHODS_TIMED:
                torch.manual_seed(SEED)
                kw = TUNED[mod] if name == "supft" else {}
                r = timed(METHODS[name](model, **kw), T, dev)
                out[f"{dev}/{mod}/{name}"] = r
                print(f"[{dev} {mod}] {name:14s} {r['ms_per_frame_amortised']:7.3f} ms/frame amortised   "
                      f"{r['ms_per_update']:8.2f} ms per labelled batch", flush=True)
            del T
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
