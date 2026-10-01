"""Paper figures, one per file, 600 dpi PNG plus vector PDF, from results/*.json only.

Every series carries a redundant non-colour encoding (marker or line style) so the figures survive a
black-and-white print. Legends sit outside the plotting area.

    python src/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "grey": "#52514e", "light": "#b5b4b0"}
NAME = {"radar": "mmWave radar", "wifi": "Wi-Fi CSI"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                     "lines.linewidth": 1.3, "lines.markersize": 4})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_budget():
    """MPJPE against the share of labelled frames: affine correction + smoothing vs tuned online fine-tuning."""
    b = [r for r in json.loads((RES / "budget.json").read_text()) if "mpjpe_mm" in r]
    h = {(r["modality"], r["method"]): r for r in json.loads((RES / "headline.json").read_text())}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    style = {"affine+smooth": (PALETTE["blue"], "-", "o", "Affine correction + smoothing"),
             "supft": (PALETTE["orange"], "--", "s", "Online fine-tuning (tuned)")}
    for ax, mod, tag in zip(axes, ("radar", "wifi"), "ab"):
        for m, (c, ls, mk, lab) in style.items():
            rows = sorted((r for r in b if r["modality"] == mod and r["method"] == m), key=lambda r: r["label_pct"])
            x = np.array([r["label_pct"] for r in rows])
            mu = np.array([r["mpjpe_mm"] for r in rows])
            sd = np.array([r["mpjpe_std"] for r in rows])
            ax.plot(x, mu, color=c, linestyle=ls, marker=mk, label=lab)
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
        ax.axhline(h[(mod, "source")]["mpjpe_mm"], color=PALETTE["grey"], linestyle=":", linewidth=1.0,
                   label="Frozen model")
        ax.axhline(h[(mod, "within-room")]["mpjpe_mm"], color=PALETTE["grey"], linestyle="-.", linewidth=1.0,
                   label="Within-room reference")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
        ax.set_xticklabels(["0.1", "", "0.5", "1", "2", "5"])
        ax.minorticks_off()
        ax.set_title(f"({tag}) {NAME[mod]}", fontsize=8)
    axes[0].set_ylabel("MPJPE (mm)")
    fig.supxlabel("labelled frames (%)", fontsize=8, y=-0.1)
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_budget")


def fig_mechanism():
    """Predicted vs true depth of the body's centre along the held-out-room stream."""
    d = json.loads((RES / "fig_mechanism.json").read_text())
    panels = (("radar_source", "(a) radar, frozen"), ("radar_norm", "(b) radar, norm. TTA"),
              ("wifi_source", "(c) Wi-Fi, frozen"))
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.45))
    for ax, (key, title) in zip(axes, panels):
        P, G = np.array(d[key]["pred_m"])[:, 2], np.array(d[key]["true_m"])[:, 2]
        ax.plot(G, P, "o", color=PALETTE["blue"], markersize=1.2, alpha=0.35, markeredgewidth=0, rasterized=True)
        lo, hi = G.min() - 0.1, G.max() + 0.1
        ax.plot([lo, hi], [lo, hi], color=PALETTE["grey"], linestyle="--", linewidth=0.8)
        ax.set_xlim(lo, hi)
        if key != "radar_norm":
            ax.set_ylim(lo, hi)
        r = np.corrcoef(G, P)[0, 1]
        ax.set_title(f"{title}\nr = {r:+.2f}", fontsize=7)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("predicted depth (m)", fontsize=7)
    fig.supxlabel("true depth (m)", fontsize=7, y=-0.08)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig_mechanism")


def graphical_abstract():
    """660 x 295 px at 300 dpi (IEEE graphical-abstract size): (a) the radar model's depth, frozen and under
    normalization TTA; (b) MPJPE at 1% labels, both sensors, against the frozen model and the within-room level."""
    d = json.loads((RES / "fig_mechanism.json").read_text())
    h = {(r["modality"], r["method"]): r for r in json.loads((RES / "headline.json").read_text())}
    with plt.rc_context({"font.size": 5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 1.5, "ytick.major.size": 1.5, "grid.linewidth": 0.3}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(2.2, 0.985), gridspec_kw={"width_ratios": [1, 1.25]})
        for key, c, lab in (("radar_norm", PALETTE["orange"], "label-free TTA"),
                            ("radar_source", PALETTE["blue"], "frozen")):
            P, G = np.array(d[key]["pred_m"])[:, 2], np.array(d[key]["true_m"])[:, 2]
            a.plot(G, P, "o", color=c, markersize=0.6, alpha=0.4, markeredgewidth=0, rasterized=True, label=lab)
        a.plot([2.6, 3.6], [2.6, 3.6], color=PALETTE["grey"], linestyle="--", linewidth=0.5)
        a.set_ylim(2, 6)
        a.set_xlim(2.6, 3.6)
        a.set_title("(a) radar depth", fontsize=5, pad=2)
        a.set_xlabel("true depth (m)", labelpad=1)
        a.set_ylabel("predicted (m)", labelpad=1)
        a.legend(loc="upper left", fontsize=3.6, markerscale=4, handletextpad=0.1, borderaxespad=0.2)
        x = np.arange(2)
        for k, (meth, c, lab, hatch) in enumerate((("source", PALETTE["grey"], "frozen", ""),
                                                    ("affine+smooth", PALETTE["blue"], "12-param. correction", ""),
                                                    ("supft", PALETTE["orange"], "fine-tuning", "////"))):
            v = [h[(m, meth)]["mpjpe_mm"] for m in ("radar", "wifi")]
            b.bar(x + (k - 1) * 0.27, v, 0.25, color=c, label=lab, hatch=hatch, edgecolor="white", linewidth=0.3)
        for j, m in enumerate(("radar", "wifi")):
            w = h[(m, "within-room")]["mpjpe_mm"]
            b.plot([j - 0.42, j + 0.42], [w, w], color="black", linewidth=0.6, linestyle=":",
                   label="same-room level" if j == 0 else None)
        b.set_xticks(x)
        b.set_xticklabels(["mmWave radar", "Wi-Fi"])
        b.set_ylabel("MPJPE (mm)", labelpad=1)
        b.set_title("(b) 1% of frames labelled", fontsize=5, pad=2)
        b.legend(loc="upper left", fontsize=3.6, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
        b.set_ylim(0, 380)
        b.grid(axis="x", visible=False)
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.24, top=0.88, wspace=0.38)
        fig.savefig(FIG / "graphical_abstract.png", dpi=300)
        plt.close(fig)
    from PIL import Image                         # palette PNG: IEEE recommends under 45 KB
    im = Image.open(FIG / "graphical_abstract.png").convert("RGB").resize((660, 295))
    im.quantize(colors=64).save(FIG / "graphical_abstract.png", optimize=True)


if __name__ == "__main__":
    fig_budget()
    fig_mechanism()
    graphical_abstract()
    print("figures written to", FIG)
