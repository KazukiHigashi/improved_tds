"""Check the paper training curves and their frozen validation provenance."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "analysis/paper_ff_fb_generalization_20261008"


def test_training_curves_are_all_real_validation_series() -> None:
    html = (REPORT / "index.html").read_text(encoding="utf-8")
    section = html.split("<section id='training-detail'>", 1)[1].split("</section>", 1)[0]
    assert "href='#training-detail'" in html
    assert section.count("class='training-chart'") == 6
    assert section.count("class='training-chart-pair'") == 3
    assert "学習中ロールアウトの報酬ではない" in section
    assert "32初期姿勢×11目標＝352試行" in section
    history = json.loads((REPORT / "training_reward_history.json").read_text(encoding="utf-8"))
    assert history["protocol"] == "paper15_validation_mean_step_reward_v1"
    assert len(history["tasks"]) == 15
    for task in history["tasks"]:
        assert len(task["points"]) == task["training_steps"] // 100_000
        assert all(point["validation_episodes"] == 352 for point in task["points"])
        assert all(math.isfinite(point["mean_reward_per_sim_step"])
                   for point in task["points"])


def test_validation_reward_is_step_weighted_not_episode_sum() -> None:
    task = "scissors_lite_1"
    recorded = json.loads((ROOT / "runs/paper15_unified_3m_slew_20260927" /
                           task / "validation_history.json").read_text(encoding="utf-8"))
    exported = json.loads((REPORT / "training_reward_history.json").read_text(encoding="utf-8"))
    point = next(row for row in exported["tasks"] if row["task"] == task)["points"][0]
    episodes = recorded["history"][0]["episode_metrics"]
    expected = sum(e["mean_reward_terms"]["reward_total"] * e["episode_steps"]
                   for e in episodes) / sum(e["episode_steps"] for e in episodes)
    assert math.isclose(point["mean_reward_per_sim_step"], expected, rel_tol=1e-12)
