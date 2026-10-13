"""The selected-paper report must not depend on archived experiments."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "analysis/paper_ff_fb_generalization_20261008"
pytestmark = pytest.mark.skipif(
    not (PAPER / "index.html").is_file(),
    reason="local paper report is intentionally not tracked by Git",
)


class LocalLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        candidate = attributes.get("href" if tag == "a" else "src" if tag == "img" else "")
        if candidate:
            self.links.append(candidate)


def test_paper_html_has_only_existing_local_links_and_old_conditions() -> None:
    text = (PAPER / "index.html").read_text(encoding="utf-8")
    assert "spray_high_angle" not in text
    assert "id='spray-retrain'" not in text
    assert "id='spray-unknown'" not in text
    assert "id='experiment-map'" in text
    for artifact in (
        "scripts/run_paper15_unified.py",
        "scripts/run_paper15_tds_corrected.py",
        "scripts/replay_matched_nine_ff.py",
        "scripts/run_second_axis_feedback.py",
        "scripts/replay_matched_nine_generalization.py",
        "scripts/build_paper_ff_fb_generalization_report.py",
        "analysis/paper15_paper_bundle_20260928/manifest.json",
        "analysis/matched_nine_goals_clipped_20261008/summary.json",
    ):
        assert artifact in text
    summary = json.loads((PAPER / "summary.json").read_text(encoding="utf-8"))
    assert "spray_retraining" not in summary
    assert "spray_unknown" not in summary
    parser = LocalLinks()
    parser.feed(text)
    missing = []
    for link in parser.links:
        parsed = urlsplit(link)
        if parsed.scheme or not parsed.path:
            continue
        target = (PAPER / unquote(parsed.path)).resolve()
        if not target.is_file():
            missing.append(link)
    assert not missing, f"Paper HTML references missing files: {missing[:15]}"
