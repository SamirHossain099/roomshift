"""Headline numbers for project 02 from results/. Writes results/headline.json, results/budget.json and
results/calonce.json and prints them.

Final runs (2026-10-01) are tagged _F2 (K=100, all methods), _B2 (label budget) and _CAL<F> (calibrate once), with
the supervised baseline at the settings tuned on all four rooms in E3b (scripts/sweep_supft.py): radar lr 3e-6 x 4
steps, Wi-Fi lr 1e-3 x 64 steps. An earlier table (tag _F1, not released) used an under-tuned baseline.

MPJPE is averaged over the four held-out rooms per seed, then mean +- std over seeds.

    python scripts/headline.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ROOMS = ("E01", "E02", "E03", "E04")
ORDER = ["source", "norm", "smooth", "rigid", "affine", "affine+smooth", "supft"]
LABELS = {"source": "source", "norm": "normalisation TTA (label-free)", "smooth": "temporal smoothing (label-free)",
          "rigid": "rigid calibration (1% labels)", "affine": "affine calibration (1% labels)",
          "affine+smooth": "affine calibration + smoothing (1% labels)",
          "supft": "supervised online fine-tuning (1% labels, tuned: radar lr 3e-6 x 4 steps, Wi-Fi lr 1e-3 x 64 steps)"}
TUNED = {"radar": ["lr=3e-6", "steps=4"], "wifi": ["lr=1e-3", "steps=64"]}
KEYS = ("mpjpe_mm", "root_err_mm", "rel_mpjpe_mm", "pa_mpjpe_mm", "act_acc")


def _runs(pattern):
    for f in sorted(R.glob(pattern)):
        r = json.loads(f.read_text())
        if r["kw"][:2] != TUNED[r["modality"]]:
            raise ValueError(f"{f.name}: supervised settings {r['kw']} are not the tuned ones")
        yield r


def _summary(by, keys=KEYS):
    """by: seed -> room -> result. Mean over rooms per seed, then mean/std over seeds (complete seeds only)."""
    seeds = sorted(s for s, v in by.items() if len(v) == len(ROOMS))
    if not seeds:
        return None
    vals = {k: [np.mean([by[s][h][k] for h in ROOMS]) for s in seeds] for k in keys}
    out = {k: round(float(np.mean(v)), 3 if k == "act_acc" else 1) for k, v in vals.items()}
    return out | {"mpjpe_std": round(float(np.std(vals["mpjpe_mm"])), 1), "seeds": len(seeds)}


def within_reference():
    ref = defaultdict(dict)  # modality -> seed -> room -> within-room MPJPE
    for f in sorted(R.glob("E0_*.json")):
        r = json.loads(f.read_text())
        for run in r["runs"]:
            ref[r["modality"]].setdefault(run["seed"], {})[run["hold"]] = run["within"]["mpjpe_mm"]
    return {m: {s: float(np.mean(list(v.values()))) for s, v in d.items() if len(v) == 4} for m, d in ref.items()}


def headline():
    by = defaultdict(lambda: defaultdict(dict))   # (modality, method) -> seed -> room
    for r in _runs("stream_*_K100_F2_*.json"):
        for m, v in r["methods"].items():
            by[(r["modality"], m)][r["seed"]][r["hold"]] = v
    ref = within_reference()
    rows = []
    for mod in ("radar", "wifi"):
        for m in ORDER:
            s = _summary(by[(mod, m)])
            if s:
                rows.append({"modality": mod, "method": m, "label": LABELS[m], **s})
        w = list(ref[mod].values())
        rows.append({"modality": mod, "method": "within-room", "label": "within-room reference (no shift)",
                     "seeds": len(w), "mpjpe_mm": round(float(np.mean(w)), 1), "mpjpe_std": round(float(np.std(w)), 1)})
    return rows


def budget():
    by = defaultdict(lambda: defaultdict(dict))   # (modality, method, K) -> seed -> room
    for pat in ("stream_*_B2_*.json", "stream_*_K100_F2_*.json"):
        for r in _runs(pat):
            for m in ("affine+smooth", "supft"):
                by[(r["modality"], m, r["label_every"])][r["seed"]][r["hold"]] = r["methods"][m]
    rows = []
    for (mod, m, K), v in sorted(by.items()):
        s = _summary(v, ("mpjpe_mm",))
        if s:
            rows.append({"modality": mod, "method": m, "label_every": K, "label_pct": 100 / K, **s})
    # paired: calibration minus fine-tuning on the same (seed, room); negative = calibration better
    for mod, K in sorted({(r["modality"], r["label_every"]) for r in rows}):
        a, b = by[(mod, "affine+smooth", K)], by[(mod, "supft", K)]
        d = [a[s][h]["mpjpe_mm"] - b[s][h]["mpjpe_mm"] for s in a if s in b for h in ROOMS if h in a[s] and h in b[s]]
        if len(d) == len(ROOMS) * 3:
            rows.append({"modality": mod, "method": "paired:affine+smooth-supft", "label_every": K, "label_pct": 100 / K,
                         "diff_mm": round(float(np.mean(d)), 1), "diff_min": round(float(min(d)), 1),
                         "diff_max": round(float(max(d)), 1), "n_calib_better": int(sum(x < 0 for x in d)), "n_pairs": len(d)})
    return rows


def calonce():
    """Fit once on the first F labels (one every 100 frames), freeze; score only the subjects that gave no label.
    Continuous updating (F=0) is scored on the same subject sets, for each frozen budget's set."""
    by = defaultdict(lambda: defaultdict(dict))   # (modality, method, F, dropped subjects) -> seed -> room
    for f in sorted(R.glob("stream_*_K100_CAL*_*.json")):
        r = json.loads(f.read_text())
        F = int(f.name.split("_CAL")[1].split("_")[0])
        for m, v in r["methods"].items():
            ps = v["per_subject"]
            subs = sorted(ps, key=int)                             # stream order is subject order
            start = np.cumsum([0] + [ps[s]["n"] for s in subs])[:-1]
            drops = [int((start < 100 * F).sum())] if F else [1, 3]
            for d in drops:
                keep = subs[d:]
                n = sum(ps[s]["n"] for s in keep)
                by[(r["modality"], m, F, d)][r["seed"]][r["hold"]] = {
                    k: sum(ps[s][k] * ps[s]["n"] for s in keep) / n for k in KEYS}
    rows = []
    for (mod, m, F, d), v in sorted(by.items()):
        s = _summary(v, ("mpjpe_mm", "root_err_mm"))
        if s:
            rows.append({"modality": mod, "method": m, "freeze_after": F, "calibration_subjects": d, **s})
    return rows


def labelfree():
    """Label-free methods on all rooms and seeds, recorded order (_LF3) and shuffled frames (_IID3)."""
    by = defaultdict(lambda: defaultdict(dict))   # (modality, order, method) -> seed -> room
    for tag, order in (("_LF3", "sequential"), ("_IID3", "shuffled")):
        for f in sorted(R.glob(f"stream_*_K0{tag}_*.json")):
            r = json.loads(f.read_text())
            for m, v in r["methods"].items():
                by[(r["modality"], order, m)][r["seed"]][r["hold"]] = v
    rows = []
    for (mod, order, m), v in sorted(by.items()):
        s = _summary(v)
        if s:
            rows.append({"modality": mod, "order": order, "method": m, **s})
    return rows


def main():
    out = {"headline": headline(), "budget": budget(), "calonce": calonce(), "labelfree": labelfree()}
    for r in out["headline"]:
        print(f"{r['modality']:6s} {r['label'][:52]:52s} n={r['seeds']}  MPJPE {r['mpjpe_mm']:7.1f} +- {r['mpjpe_std']:.1f}"
              + (f"  root {r['root_err_mm']:7.1f}  body {r['rel_mpjpe_mm']:6.1f}  act {r['act_acc']:.3f}" if "root_err_mm" in r else ""))
    print("\nlabel budget (MPJPE, mean over rooms and seeds)")
    for r in out["budget"]:
        if "diff_mm" in r:
            print(f"  {r['modality']:6s} paired calib-FT {r['label_pct']:5.2f}%  {r['diff_mm']:+6.1f} mm "
                  f"[{r['diff_min']:+.1f}, {r['diff_max']:+.1f}]  calibration better in {r['n_calib_better']}/{r['n_pairs']}")
            continue
        print(f"  {r['modality']:6s} {r['method']:14s} {r['label_pct']:5.2f}%  {r['mpjpe_mm']:6.1f} +- {r['mpjpe_std']:.1f}  n={r['seeds']}")
    print("\ncalibrate once (scored on subjects that gave no label; F=0 = continuous)")
    for r in out["calonce"]:
        print(f"  {r['modality']:6s} {r['method']:14s} F={r['freeze_after']:3d} skip {r['calibration_subjects']}  "
              f"{r['mpjpe_mm']:6.1f} +- {r['mpjpe_std']:.1f}  "
              f"root {r['root_err_mm']:6.1f}  n={r['seeds']}")
    (R / "headline.json").write_text(json.dumps(out["headline"], indent=1))
    (R / "budget.json").write_text(json.dumps(out["budget"], indent=1))
    (R / "calonce.json").write_text(json.dumps(out["calonce"], indent=1))
    (R / "labelfree.json").write_text(json.dumps(out["labelfree"], indent=1))
    print("\nlabel-free")
    for r in out["labelfree"]:
        print(f"  {r['modality']:6s} {r['order']:10s} {r['method']:8s} MPJPE {r['mpjpe_mm']:7.1f} +- {r['mpjpe_std']:.1f}  "
              f"root {r['root_err_mm']:7.1f}  body {r['rel_mpjpe_mm']:6.1f}  act {r['act_acc']:.3f}  n={r['seeds']}")
    print("wrote results/headline.json, budget.json, calonce.json")


if __name__ == "__main__":
    main()
