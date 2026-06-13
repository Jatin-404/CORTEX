"""Golden evaluation question set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cortex.settings import Settings


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    ground_truth: str
    expected_paths: list[str]


def load_golden_questions(path: Path | None = None) -> list[GoldenQuestion]:
    settings = Settings()
    golden_path = path or settings.eval_golden_path
    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    items: list[GoldenQuestion] = []
    for entry in raw:
        items.append(
            GoldenQuestion(
                id=entry["id"],
                question=entry["question"],
                ground_truth=entry.get("ground_truth", ""),
                expected_paths=list(entry.get("expected_paths", [])),
            )
        )
    return items
