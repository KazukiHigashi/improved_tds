"""Lightweight checks for the frozen full-RL nine-target replay rules."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
from statistics import mean
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nine_targets_include_rl_endpoints():
    ff = load("replay_matched_nine_ff")
    unknown = load("replay_matched_nine_generalization")
    assert len(ff.FRACTIONS) == len(unknown.FRACTIONS) == 9
    assert np.allclose(ff.FRACTIONS, np.linspace(0., 1., 9))
    assert ff.FRACTIONS == unknown.FRACTIONS


def test_frozen_ff_inverse_clips_to_admitted_domain_not_outer_knots():
    ff = load("replay_matched_nine_ff")
    config = dict(selection=dict(map_tool=[-.1, .2, .8, 1.1],
                                 map_u=[-1., -.6, .6, 1.],
                                 rho_start=-2., rho_end=2.),
                  inverse_target_range=[.2, .8])
    below, clipped_below = ff.frozen_score(config, 0.)
    inside, clipped_inside = ff.frozen_score(config, .5)
    above, clipped_above = ff.frozen_score(config, 1.)
    assert np.isclose(below, -.6 * 2.)
    assert np.isclose(inside, 0.)
    assert np.isclose(above, .6 * 2.)
    assert clipped_below and clipped_above and not clipped_inside


def test_unknown_zero_score_uses_normalized_fraction_not_physical_units():
    unknown = load("replay_matched_nine_generalization")
    package = (np.zeros(13), np.ones(13), np.array([.5, .25]),
               np.array([-2., 2.]))
    score, clipped = unknown.command_score(.75, .006, package, None)
    assert np.isclose(score, 1.)
    assert not clipped


def test_mae_illustration_is_first_frozen_pose_and_matches_raw_trials():
    report = load("build_matched_nine_report")
    result = report.RESULT
    source = json.loads((result / "ff" / "scissors_lite_3.json").read_text(encoding="utf-8"))
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    example = summary["mae_illustration"]
    assert example["pose"] == source["test_poses"][0]
    assert example["method"] == "pls"
    assert len(example["rows"]) == 9
    span = source["target_range"][1] - source["target_range"][0]
    for row in example["rows"]:
        assert np.isclose(row["normalized_abs_error"],
                          abs(row["achieved_angle"] - row["target_angle"]) / span)
    assert np.isclose(example["nine_target_mae_percent"],
                      100 * mean(row["normalized_abs_error"] for row in example["rows"]))
    assert "図B" in report.mae_illustration(example)


def test_paper_report_contains_visual_guide_directly():
    page = (ROOT / "analysis" / "paper_ff_fb_generalization_20261008" / "index.html")
    text = page.read_text(encoding="utf-8")
    assert "id='visual-guide'" in text
    assert "href='#visual-guide'" in text
    assert "class='procedure-figure'" in text
    assert "class='mae-svg'" in text
    assert "従来の主FFは方式ごとの逆写像可能域内で9目標" in text


def test_paper_report_distinguishes_physical_and_db_inverse_protocols():
    page = (ROOT / "analysis" / "paper_ff_fb_generalization_20261008" / "index.html")
    text = page.read_text(encoding="utf-8")
    assert "href='#protocol-comparison'" in text
    section = text.split("<section id='protocol-comparison'>", 1)[1].split("</section>", 1)[0]
    assert "4.1節：自ツールDB・線形逆写像参照" in section
    assert "較正0姿勢／評価3姿勢" in section
    assert "17スコア×8姿勢" in section
    assert "3～15プローブ" in section
    assert "z_{\\mathrm{DB}}" in section
    assert "z_{\\mathrm{FF}}" in section
    assert "A_{\\mathrm{held}}" in section
    assert "A_{\\mathrm{finite}}" in section
    assert "0.4998 rad" in section and "0.1863 rad" in section
    assert "再学習・再フィットなし" not in section


def test_paper_report_embeds_matched_results_and_closed_representative_gallery():
    page = ROOT / "analysis" / "paper_ff_fb_generalization_20261008" / "index.html"
    text = page.read_text(encoding="utf-8")
    assert "href='#matched-nine'" in text
    assert "href='#representatives'" in text
    matched = text.split("<section id='matched-nine'>", 1)[1].split("</section>", 1)[0]
    assert "id='matched-ff'" in matched
    assert "id='matched-unknown'" in matched
    assert "id='matched-fb'" in matched
    assert "id='matched-coverage'" in matched
    assert "全15ツール・6方式の数値" in matched
    gallery = text.split("<section id='representatives'>", 1)[1].split("</section>", 1)[0]
    assert gallery.count("<details>") == 36
    assert "<details open" not in gallery
    manifest_root = page.parent / "representative_frames_nonzero"
    manifest = json.loads((manifest_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "matched_nine_ranked_representative_frames_nonzero_v1"
    assert len(manifest["cases"]) == 36
    assert "目標値0を除いた" in gallery
    assert gallery.count("<img src='representative_frames_nonzero/") == sum(
        len(case["frames"]) for case in manifest["cases"])
    for case in manifest["cases"]:
        assert abs(case["target"]) > 1e-12
        assert case["replay_abs_difference"] < (1e-4 if case["task"].startswith("push") else 5e-3)
        assert case["replay_held"] == case["held"]
        assert case["frames"][0]["step"] == 0
        assert all(frame["camera_dot_palm_normal"] <= -.999 for frame in case["frames"])
        assert all(frame["step"] % 25 == 0 and (manifest_root / frame["image"]).is_file()
                   for frame in case["frames"])


def test_representative_ranks_are_selected_from_nonzero_evaluation_pool():
    gallery = load("render_matched_representatives")
    root = ROOT / "analysis" / "paper_ff_fb_generalization_20261008" / "representative_frames_nonzero"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert all(gallery.nonzero_target(case["target"]) for case in manifest["cases"])
    selected = gallery.select_cases(("ff_pca", "unknown_selected_three_probe", "fb_pls_axis2_fb"))
    actual = {(case["group"], case["rank"]): case for case in manifest["cases"]}
    for case in selected:
        saved = actual[(case["group"], case["rank"])]
        assert saved["task"] == case["task"]
        assert saved["pose"] == case["pose"]
        assert np.isclose(saved["target"], case["target"])
        assert np.isclose(saved["error_percent"], case["error_percent"])


def test_paper_report_shows_all_source_verified_hand_only_synergy_sweeps():
    page = ROOT / "analysis" / "paper_ff_fb_generalization_20261008" / "index.html"
    text = page.read_text(encoding="utf-8")
    assert "href='#synergy-sweeps'" in text
    section = text.split("<section id='synergy-sweeps'>", 1)[1].split("</section>", 1)[0]
    assert section.count("<details>") == 15
    assert "<details open" not in section
    assert "PLS第2潜在成分ではなく" in section
    assert section.count("class='sweep-frame'") == 300
    sources = re.findall(r"<img src='([^']+)'", section)
    assert len(sources) == len(set(sources)) == 300
    assert all((page.parent / source).is_file() for source in sources)
    generated = load("build_paper_ff_fb_generalization_report").synergy_sweep_section()
    assert f"<section id='synergy-sweeps'>{section}</section>" == generated
