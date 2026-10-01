"""Pin the paper's claims to results/. Run: python -m pytest tests/ -q

Every assertion reads results/*.json written by scripts/headline.py and the analysis scripts; none hardcodes the
number it guards. Each claim below is one the manuscript makes in words.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def _j(name):
    return json.loads((R / name).read_text())


def _h(mod, method):
    return next(r for r in _j("headline.json") if r["modality"] == mod and r["method"] == method)


def _paired(mod):
    return [r for r in _j("budget.json") if r["modality"] == mod and r["method"].startswith("paired")]


def test_aggregates_are_recomputable_and_complete():
    before = {n: _j(n) for n in ("headline.json", "budget.json", "calonce.json", "labelfree.json")}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "headline.py")], check=True, capture_output=True)
    assert {n: _j(n) for n in before} == before
    assert all(r["seeds"] == 3 for r in before["headline.json"])
    assert all(r.get("seeds", 3) == 3 for r in before["budget.json"])
    assert all(r["seeds"] == 3 for r in before["labelfree.json"])


def test_shift_is_mostly_root():
    m = _j("manuscript_numbers.json")
    for mod in ("radar", "wifi"):
        assert m[f"e0_root_growth_{mod}"]["value"] > 2 * m[f"e0_body_growth_{mod}"]["value"]


def test_placement_test_significant():
    assert _j("confound.json")["placement"]["perm_p"] <= 0.001


def test_normalization_tta_hurts_and_helps_action_when_shuffled():
    lf = {(r["modality"], r["order"], r["method"]): r for r in _j("labelfree.json")}
    for mod in ("radar", "wifi"):
        assert lf[(mod, "sequential", "norm")]["mpjpe_mm"] > lf[(mod, "sequential", "source")]["mpjpe_mm"]
    assert lf[("radar", "sequential", "norm")]["mpjpe_mm"] > 5 * lf[("radar", "sequential", "source")]["mpjpe_mm"]
    # shuffled: root error still rises, action accuracy rises
    assert lf[("radar", "shuffled", "norm")]["root_err_mm"] > lf[("radar", "shuffled", "source")]["root_err_mm"]
    assert lf[("radar", "shuffled", "norm")]["act_acc"] > lf[("radar", "shuffled", "source")]["act_acc"]


def test_radar_statistics_carry_position_wifi_does_not_localize_in_new_room():
    m = _j("manuscript_numbers.json")
    assert m["probe_r2_radar_min"]["value"] > 0.5
    assert m["corr_src_radar_min"]["value"] > 0.6
    assert abs(m["corr_src_wifi_min"]["value"]) < 0.25 and abs(m["corr_src_wifi_max"]["value"]) < 0.25
    assert m["corr_norm_z_max"]["value"] < 0.1


def test_radar_calibration_equivalent_to_tuned_finetuning():
    # "matches ... at every label rate from 0.1% to 2%": mean paired difference within 3 mm, neither side consistent
    for r in _paired("radar"):
        if r["label_every"] >= 50:
            assert abs(r["diff_mm"]) < 3.0
            assert 3 <= r["n_calib_better"] <= r["n_pairs"] - 3
    assert abs(_h("radar", "affine+smooth")["mpjpe_mm"] - _h("radar", "supft")["mpjpe_mm"]) < 1.0


def test_wifi_finetuning_better_at_every_budget():
    for r in _paired("wifi"):
        assert r["diff_mm"] > 0 and r["n_calib_better"] == 0
    assert _h("wifi", "supft")["mpjpe_mm"] < _h("wifi", "within-room")["mpjpe_mm"]


def test_calibrate_once_keeps_little_on_radar_more_on_wifi():
    m = _j("manuscript_numbers.json")
    assert m["once_share_radar"]["value"] < 25
    assert m["once_share_wifi"]["value"] > 50
    assert m["once1_radar"]["value"] > m["once_src1_radar"]["value"]      # one-subject fit worse than none


def test_every_correction_beats_frozen_model():
    for mod in ("radar", "wifi"):
        for meth in ("rigid", "affine", "affine+smooth", "supft"):
            assert _h(mod, meth)["mpjpe_mm"] < _h(mod, "source")["mpjpe_mm"]
