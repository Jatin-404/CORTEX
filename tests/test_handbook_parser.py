from pathlib import Path

import pytest

from cortex.ingestion.parsers.handbook import HandbookParser
from cortex.models.enums import SourceType


@pytest.fixture
def handbook_root() -> Path:
    root = Path("data/handbook-main/handbook-main/content/handbook")
    if not root.exists():
        pytest.skip("handbook data not available locally")
    return root


def test_parse_index_page(handbook_root: Path) -> None:
    parser = HandbookParser(handbook_root)
    doc = parser.parse_file(handbook_root / "values" / "_index.md")
    assert doc is not None
    assert doc.source_type == SourceType.HANDBOOK_MARKDOWN
    assert doc.metadata["department"] == "values"
    assert doc.metadata["is_index"] is True
    assert len(doc.sections) > 0


def test_parse_nested_page(handbook_root: Path) -> None:
    parser = HandbookParser(handbook_root)
    doc = parser.parse_file(handbook_root / "about" / "contributing.md")
    assert doc is not None
    assert doc.metadata["department"] == "about"
    assert doc.doc_id  # deterministic


def test_iter_documents_yields(handbook_root: Path) -> None:
    parser = HandbookParser(handbook_root)
    doc = next(parser.iter_documents(), None)
    assert doc is not None
    assert doc.source_type == SourceType.HANDBOOK_MARKDOWN
