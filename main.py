"""Entry point — delegates to the ingestion CLI."""

from cortex.cli.ingest import main

if __name__ == "__main__":
    raise SystemExit(main())
