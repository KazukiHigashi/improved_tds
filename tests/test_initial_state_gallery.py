"""Integrity checks for the 15-task initial-state image gallery."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "analysis/paper_ff_fb_generalization_20261008"
pytestmark = pytest.mark.skipif(
    not (REPORT / "index.html").is_file(),
    reason="local paper report is intentionally not tracked by Git",
)


def test_all_initial_states_are_paired_and_source_verified() -> None:
    page = (REPORT / "index.html").read_text(encoding="utf-8")
    assert "href='#initial-states'" in page
    section = page.split("<section id='initial-states'>", 1)[1].split("</section>", 1)[0]
    assert section.count("<details>") == 3
    assert "<details open" not in section
    assert section.count("class='initial-task'") == 15
    images = re.findall(r"<img src='([^']+)'", section)
    assert len(images) == len(set(images)) == 30
    assert all((REPORT / image).is_file() for image in images)
    manifest = json.loads((REPORT / "initial_states/manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "frozen_ff_test_initial_state_palm_facing_v1"
    assert manifest["no_simulation_step"]
    assert len(manifest["tasks"]) == 15
    for item in manifest["tasks"]:
        assert item["pool_size"] == 32
        assert len(item["initial_actuated_q"]) == 13
        assert item["camera_dot_palm_normal"] <= -.999
        assert (REPORT / "initial_states" / item["state_image"]).is_file()
        assert (REPORT / "initial_states" / item["hand_image"]).is_file()
