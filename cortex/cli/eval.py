"""CLI for golden-set evaluation (structural checks + optional RAGAS)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cortex.eval.runner import run_eval, save_report
from cortex.logging_config import configure_logging
from cortex.settings import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Cortex KB against golden questions")
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Path to golden JSON (default: eval/golden_handbook.json)",
    )
    parser.add_argument("-n", "--limit", type=int, help="Evaluate only first N questions")
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS metrics (structural checks only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full report JSON to stdout",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress info logs")

    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else settings.log_level)

    try:
        report = run_eval(
            settings,
            golden_path=args.golden,
            use_ragas=False if args.no_ragas else None,
            limit=args.limit,
        )
        out_path = save_report(report)
    except Exception as exc:
        logging.getLogger(__name__).error("eval_failed: %s", exc)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Eval run: {report.run_id}")
        print(f"Questions: {report.question_count}")
        print(f"Path hit rate: {report.path_hit_rate:.0%}")
        print(f"Grade pass rate: {report.grade_pass_rate:.0%}")
        if report.ragas_averages:
            print("RAGAS averages:")
            for name, value in sorted(report.ragas_averages.items()):
                print(f"  {name}: {value:.3f}")
        print(f"Report: {out_path}")
        for result in report.results:
            status = "PASS" if result.path_hit and result.grade_passed else "FAIL"
            print(f"  [{status}] {result.id}: paths={result.source_paths[:2]}")

    failed = [r for r in report.results if not r.path_hit or not r.grade_passed]
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
