"""CLI entry point for handbook ingestion."""

from __future__ import annotations

import argparse
import logging
import sys

from cortex.ingestion.pipeline import IngestionPipeline
from cortex.logging_config import configure_logging
from cortex.settings import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cortex ingestion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    handbook_parser = sub.add_parser("handbook", help="Ingest GitLab handbook markdown")
    handbook_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all documents even if file hash unchanged",
    )
    handbook_parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create Postgres tables and Qdrant collection only",
    )

    file_parser = sub.add_parser("file", help="Ingest a single markdown file")
    file_parser.add_argument("path", help="Absolute or relative path to .md file")

    args = parser.parse_args(argv)
    configure_logging(settings.log_level)

    pipeline = IngestionPipeline(settings)
    pipeline.initialize()

    if args.command == "handbook":
        if args.init_only:
            logging.getLogger(__name__).info("initialization_complete")
            return 0
        result = pipeline.ingest_handbook(force=args.force)
        print(
            f"Ingestion complete — job={result.job_id} "
            f"processed={result.documents_processed} "
            f"skipped={result.documents_skipped} "
            f"chunks={result.chunks_upserted} "
            f"errors={len(result.errors)}"
        )
        return 1 if result.errors else 0

    if args.command == "file":
        from pathlib import Path

        path = Path(args.path)
        chunks = pipeline.ingest_file(path, force=True)
        print(f"Ingested {chunks} chunks from {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
