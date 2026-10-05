"""Every number quoted in the paper, computed from results/.

Writes results/manuscript_numbers.json: {name: {"value": float, "text": formatted string, "source": file}}, with the
tables as Markdown under "table1".."table3". The manuscript text is rendered from this file, so its numbers cannot
drift from results/. A part whose results are not on disk yet is
skipped and its tokens stay unresolved, which makes the render fail loudly.

    python scripts/headline.py && python scripts/manuscript_numbers.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ROOMS = ("E01", "E02", "E03", "E04")
MODS = ("radar", "wifi")
NAME = {"radar": "mmWave radar", "wifi": "Wi-Fi CSI"}
OUT = {}


def put(name, value, fmt, source):
    OUT[name] = {"value": float(value), "text": fmt.format(value), "source": source}


def table(name, header, rows, source):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    OUT[name] = {"value": 0.0, "text": "\n".join(md), "source": source}


def _mean_over(by, key):
    """by: seed -> room -> result dict. Mean over rooms per seed, then over seeds."""
    seeds = [s for s in by if len(by[s]) == len(ROOMS)]
    return float(np.mean([np.mean([by[s][h][key] for h in ROOMS]) for s in seeds])), len(seeds)


def e0():
    by = defaultdict(lambda: defaultdict(dict))   # (modality, part) -> seed -> room
    for f in sorted(R.glob("E0_*.json")):
        r = json.loads(f.read_text())
        for run in r["runs"]:
            for part in ("within", "cross"):
                by[(r["modality"], part)][run["seed"]][run["hold"]] = run[part]
    v = {}
    for m in MODS:
        for p in ("within", "cross"):
            for k in ("mpjpe_mm", "root_err_mm", "rel_mpjpe_mm"):
                v[(m, p, k)] = _mean_over(by[(m, p)], k)[0]
        put(f"e0_root_growth_{m}", v[(m, "cross", "root_err_mm")] - v[(m, "within", "root_err_mm")], "{:.0f}", "E0")
        put(f"e0_body_growth_{m}", v[(m, "cross", "rel_mpjpe_mm")] - v[(m, "within", "rel_mpjpe_mm")], "{:.0f}", "E0")
    put("seeds", len(by[("radar", "cross")]), "{:d}", "E0")
    # one row per error, sensors and rooms across: fits one IEEE column
    rows = [[lab] + [f"{v[(m, p, k)]:.0f}" for m in MODS for p in ("within", "cross")]
            for k, lab in (("mpjpe_mm", "MPJPE"), ("root_err_mm", "Root"), ("rel_mpjpe_mm", "Root-centred"))]
    table("table1", ["Error", "Radar, same room", "Radar, new room", "Wi-Fi, same room", "Wi-Fi, new room"], rows, "E0")


def data_facts():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from mmfi.data import load_env
    n, pts = 0, []
    for e in ROOMS:
        d = load_env(e)
        n += len(d["pose"])
        pts.append(d["n_pts"].astype(float))
    put("frames_total", n, "{:,d}", "data/cache")
    put("pts_mean", np.concatenate(pts).mean(), "{:.0f}", "data/cache")


def confound():
    c = json.loads((R / "confound.json").read_text())
    for i, ax in enumerate("xyz"):
        put(f"conf_between_{ax}", c["between_room_sd_mm"][i], "{:.0f}", "confound")
        put(f"conf_within_{ax}", c["within_room_between_subject_sd_mm"][i], "{:.0f}", "confound")
        put(f"conf_resid0_{ax}", c["placement"]["resid_sd_shared_mm"][i], "{:.0f}", "confound")
        put(f"conf_resid1_{ax}", c["placement"]["resid_sd_room_offset_mm"][i], "{:.0f}", "confound")
    p = c["placement"]["perm_p"]
    OUT["perm_p"] = {"value": p, "text": "0.0001" if p <= 1e-4 + 1e-12 else f"{p:.4f}", "source": "confound"}
    off = np.array(list(c["placement"]["room_offset_mm"].values()))
    put("conf_offset_max", np.abs(off).max(), "{:.0f}", "confound")


def mechanism():
    for m in MODS:
        d = json.loads((R / f"mechanism_{m}.json").read_text())
        last = [d[h]["probe_r2_xyz_per_bn_layer"][-1] for h in ROOMS]
        src = [c for h in ROOMS for c in d[h]["source"]["root_corr_xyz"]]
        put(f"probe_r2_{m}_min", min(min(x) for x in last), "{:.2f}", f"mechanism_{m}")
        put(f"probe_r2_{m}_max", max(max(x) for x in last), "{:.2f}", f"mechanism_{m}")
        put(f"corr_src_{m}_min", min(src), "{:+.2f}", f"mechanism_{m}")
        put(f"corr_src_{m}_max", max(src), "{:+.2f}", f"mechanism_{m}")
        if m == "radar":
            z = [d[h]["norm"]["root_corr_xyz"][2] for h in ROOMS]
            put("corr_norm_z_min", min(z), "{:+.2f}", "mechanism_radar")
            put("corr_norm_z_max", max(z), "{:+.2f}", "mechanism_radar")
            sd = [d[h]["norm"]["pred_root_sd_mm"][2] for h in ROOMS]
            tr = [d[h]["source"]["true_root_sd_mm"][2] for h in ROOMS]
            put("norm_depth_sd_min", min(sd), "{:.0f}", "mechanism_radar")
            put("norm_depth_sd_max", max(sd), "{:.0f}", "mechanism_radar")
            put("true_depth_sd_min", min(tr), "{:.0f}", "mechanism_radar")
            put("true_depth_sd_max", max(tr), "{:.0f}", "mechanism_radar")


def headline():
    h = {(r["modality"], r["method"]): r for r in json.loads((R / "headline.json").read_text())}
    rows = []
    for m in MODS:
        if (m, "supft") not in h:
            continue
        w = h[(m, "within-room")]["mpjpe_mm"]
        src = h[(m, "source")]
        put(f"within_{m}", w, "{:.0f}", "headline")
        put(f"src_{m}", src["mpjpe_mm"], "{:.0f}", "headline")
        put(f"src_root_{m}", src["root_err_mm"], "{:.0f}", "headline")
        put(f"src_act_{m}", src["act_acc"], "{:.2f}", "headline")
        put(f"cal_{m}", h[(m, "affine+smooth")]["mpjpe_mm"], "{:.1f}", "headline")
        put(f"ft_{m}", h[(m, "supft")]["mpjpe_mm"], "{:.1f}", "headline")
        put(f"cal_close_{m}", 100 * (src["mpjpe_mm"] - h[(m, "affine+smooth")]["mpjpe_mm"]) / (src["mpjpe_mm"] - w),
            "{:.0f}", "headline")
        rows.append([NAME[m], "", "", ""])               # group row: the sensor, instead of a column
        for meth, lab in (("source", "Frozen model"), ("rigid", "Rigid correction"), ("affine", "Affine correction"),
                          ("affine+smooth", "Affine + smoothing"), ("supft", "Fine-tuning (tuned)")):
            r = h[(m, meth)]
            rows.append([lab, f"{r['mpjpe_mm']:.1f} ± {r['mpjpe_std']:.1f}", f"{r['root_err_mm']:.1f}",
                         f"{r['rel_mpjpe_mm']:.1f}"])
        rows.append(["Same-room reference", f"{w:.1f}", "", ""])
    if rows:
        table("table3", ["Method", "MPJPE", "Root", "Root-centred"], rows, "headline")


def budget():
    b = json.loads((R / "budget.json").read_text())
    for m in MODS:
        pr = [r for r in b if r["modality"] == m and r["method"].startswith("paired") and r["label_every"] >= 50]
        if len(pr) < 5:
            continue
        put(f"budget_{m}_diff_min", min(r["diff_mm"] for r in pr), "{:+.1f}", "budget")
        put(f"budget_{m}_diff_max", max(r["diff_mm"] for r in pr), "{:+.1f}", "budget")
        put(f"budget_{m}_npairs_min", min(r["n_calib_better"] for r in pr), "{:d}", "budget")
        put(f"budget_{m}_npairs_max", max(r["n_calib_better"] for r in pr), "{:d}", "budget")
        put("budget_npairs", pr[0]["n_pairs"], "{:d}", "budget")
        k20 = next(r for r in b if r["modality"] == m and r["method"].startswith("paired") and r["label_every"] == 20)
        k1000 = next(r for r in b if r["modality"] == m and r["method"].startswith("paired") and r["label_every"] == 1000)
        put(f"budget_{m}_5pct", k20["diff_mm"], "{:.1f}", "budget")   # calibration minus FT: positive = FT ahead
        put(f"budget_{m}_01pct", k1000["diff_mm"], "{:.1f}", "budget")
        allp = [r for r in b if r["modality"] == m and r["method"].startswith("paired")]
        put(f"budget_{m}_calib_better_total", sum(r["n_calib_better"] for r in allp), "{:d}", "budget")
        put(f"budget_{m}_pairs_total", sum(r["n_pairs"] for r in allp), "{:d}", "budget")


def sci(x):
    """3e-07 -> 3 × 10⁻⁷ (Unicode superscripts, as the manuscript writes powers of ten)."""
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    m, e = f"{x:.0e}".split("e")
    e = str(int(e)).translate(sup)
    return f"10{e}" if m == "1" else f"{m} × 10{e}"


def sweep():
    def grid(m):
        recs = json.loads((R / f"sweep_supft_{m}_s0_K100.json").read_text())
        cells = {(r["lr"], r["steps"]) for r in recs if r["part"] == "all"}
        lrs = sorted({lr for lr, _ in cells})
        steps = sorted({s for _, s in cells})
        return recs, lrs, steps
    for m in MODS:
        recs, lrs, steps = grid(m)
        n = len({(r["lr"], r["steps"]) for r in recs if r["part"] == "all"})
        OUT[f"ft_grid_{m}"] = {"value": float(n), "source": "sweep_supft",
                               "text": f"{n} settings with learning rates from {sci(lrs[0])} to {sci(lrs[-1])} and "
                                       f"{steps[0]} to {steps[-1]} steps"}
    recs = json.loads((R / "sweep_supft_wifi_s0_K100.json").read_text())
    mean = defaultdict(list)
    for r in recs:
        mean[(r["part"], r["lr"], r["steps"])].append(r["mpjpe_mm"])
    one = min(np.mean(v) for (p, lr, s), v in mean.items() if p == "all" and s == 1 and len(v) == 4)
    best = min(np.mean(v) for (p, lr, s), v in mean.items() if p == "all" and len(v) == 4)
    put("ft_onestep_gap_wifi", one - best, "{:.0f}", "sweep_supft")


def calonce():
    c = {(r["modality"], r["method"], r["freeze_after"], r["calibration_subjects"]): r
         for r in json.loads((R / "calonce.json").read_text())}
    for m in MODS:
        for key, F, d in (("once1", 64, 1), ("once3", 240, 3), ("cont3", 0, 3), ("cont1", 0, 1)):
            put(f"{key}_{m}", c[(m, "affine+smooth", F, d)]["mpjpe_mm"], "{:.1f}", "calonce")
        put(f"once_src1_{m}", c[(m, "source", 0, 1)]["mpjpe_mm"], "{:.1f}", "calonce")
        put(f"once_src3_{m}", c[(m, "source", 0, 3)]["mpjpe_mm"], "{:.1f}", "calonce")
        gain_once = c[(m, "source", 0, 3)]["mpjpe_mm"] - c[(m, "affine+smooth", 240, 3)]["mpjpe_mm"]
        gain_cont = c[(m, "source", 0, 3)]["mpjpe_mm"] - c[(m, "affine+smooth", 0, 3)]["mpjpe_mm"]
        put(f"once_share_{m}", 100 * gain_once / gain_cont, "{:.0f}", "calonce")
        put(f"once3_shift_{m}", c[(m, "shift", 240, 3)]["mpjpe_mm"], "{:.1f}", "calonce")
    for m in MODS:
        b = json.loads((R / f"bias_{m}.json").read_text())["summary"]
        put(f"bias_room_{m}", b["room_bias_norm_mm_mean"], "{:.0f}", f"bias_{m}")
        put(f"bias_subj_{m}", b["between_subject_sd_norm_mm_mean"], "{:.0f}", f"bias_{m}")
        for i, ax in enumerate("xyz"):
            put(f"bias_slope_{m}_{ax}", -100 * b["slope_mean_xyz"][i], "{:.0f}", f"bias_{m}")   # percent


def labelfree():
    lf = {(r["modality"], r["order"], r["method"]): r for r in json.loads((R / "labelfree.json").read_text())}
    if any(lf[k]["seeds"] != 3 for k in lf):
        raise FileNotFoundError("labelfree.json: not all seeds")
    rows = []
    for m in MODS:
        for meth in ("norm", "tent", "smooth"):
            put(f"{meth}_{m}", lf[(m, "sequential", meth)]["mpjpe_mm"], "{:.0f}", "labelfree")
        put(f"iid_norm_{m}", lf[(m, "shuffled", "norm")]["mpjpe_mm"], "{:.0f}", "labelfree")
        put(f"iid_norm_root_{m}", lf[(m, "shuffled", "norm")]["root_err_mm"], "{:.0f}", "labelfree")
        put(f"iid_norm_act_{m}", lf[(m, "shuffled", "norm")]["act_acc"], "{:.2f}", "labelfree")
        put(f"iid_tent_act_{m}", lf[(m, "shuffled", "tent")]["act_acc"], "{:.2f}", "labelfree")
        rows.append([NAME[m], "", "", "", ""])          # group row: the sensor, instead of a column
        for order, meth, lab in (("sequential", "source", "Frozen model"), ("sequential", "norm", "Norm. statistics"),
                                 ("sequential", "tent", "Tent"), ("sequential", "smooth", "Smoothing"),
                                 ("shuffled", "norm", "Norm. statistics, shuffled"),
                                 ("shuffled", "tent", "Tent, shuffled")):
            r = lf[(m, order, meth)]
            rows.append([lab, f"{r['mpjpe_mm']:.0f}", f"{r['root_err_mm']:.0f}", f"{r['rel_mpjpe_mm']:.0f}",
                         f"{r['act_acc']:.2f}"])
    table("table2", ["Method", "MPJPE", "Root", "Root-centred", "Action"], rows, "labelfree")


def predeploy():
    d = json.loads((R / "predeploy.json").read_text())
    rad = [c for r in d if r["modality"] == "radar" for c in r["within_room_root_corr_xyz"]]
    wx = [r["within_room_root_corr_xyz"][0] for r in d if r["modality"] == "wifi"]
    wyz = [c for r in d if r["modality"] == "wifi" for c in r["within_room_root_corr_xyz"][1:]]
    put("pre_radar_min", min(rad), "{:.2f}", "predeploy")
    put("pre_radar_max", max(rad), "{:.2f}", "predeploy")
    put("pre_wifi_x_min", min(wx), "{:.2f}", "predeploy")
    put("pre_wifi_x_max", max(wx), "{:.2f}", "predeploy")
    put("pre_wifi_yz_max", max(wyz), "{:.2f}", "predeploy")


def latency():
    L = json.loads((R / "latency.json").read_text())
    corr = ("shift", "rigid", "affine", "affine+smooth")
    for dev in ("cuda", "cpu"):
        d = "gpu" if dev == "cuda" else "cpu"
        put(f"lat_{d}_corr_upd_max", max(L[f"{dev}/{m}/{c}"]["ms_per_update"] for m in MODS for c in corr), "{:.1f}", "latency")
        for m in MODS:
            put(f"lat_{d}_ft_upd_{m}", L[f"{dev}/{m}/supft"]["ms_per_update"], "{:,.0f}", "latency")
            put(f"lat_{d}_ft_frame_{m}", L[f"{dev}/{m}/supft"]["ms_per_frame_amortised"], "{:.1f}", "latency")
            put(f"lat_{d}_src_frame_{m}", L[f"{dev}/{m}/source"]["ms_per_frame_amortised"], "{:.1f}", "latency")
            put(f"lat_{d}_cal_frame_{m}", L[f"{dev}/{m}/affine+smooth"]["ms_per_frame_amortised"], "{:.1f}", "latency")


def main():
    for part in (e0, data_facts, confound, mechanism, headline, budget, sweep, calonce, labelfree, predeploy, latency):
        try:
            part()
        except FileNotFoundError as e:
            print(f"skipped {part.__name__}: {e.filename}")
    (R / "manuscript_numbers.json").write_text(json.dumps(OUT, indent=1))
    print(f"wrote results/manuscript_numbers.json ({len(OUT)} entries)")


if __name__ == "__main__":
    main()
