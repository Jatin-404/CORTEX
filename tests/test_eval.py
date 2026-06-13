"""Tests for golden eval dataset and structural checks."""

from __future__ import annotations

from pathlib import Path

from cortex.eval.dataset import load_golden_questions
from cortex.eval.runner import _path_hit


def test_load_golden_questions() -> None:
    path = Path("eval/golden_handbook.json")
    items = load_golden_questions(path)
    assert len(items) >= 1
    assert items[0].id == "contributing"
    assert items[0].expected_paths


def test_path_hit_matches_substring() -> None:
    assert _path_hit(
        ["about/contributing.md"],
        ["handbook/about/contributing.md"],
    )
    assert not _path_hit(
        ["about/contributing.md"],
        ["marketing/utm-strategy.md"],
    )


def test_path_hit_empty_expected_is_pass() -> None:
    assert _path_hit([], ["any/path.md"])
