from pathlib import Path

import pytest

from cortex.ingestion.chunking.parent_child import ParentChildChunker
from cortex.ingestion.parsers.handbook import HandbookParser
from cortex.models.enums import ChunkRole


@pytest.fixture
def sample_document():
    root = Path("data/handbook-main/handbook-main/content/handbook")
    if not root.exists():
        pytest.skip("handbook data not available locally")
    parser = HandbookParser(root)
    doc = parser.parse_file(root / "about" / "contributing.md")
    assert doc is not None
    return doc


def test_chunk_document_produces_children(sample_document) -> None:
    chunker = ParentChildChunker(child_chunk_tokens=100, parent_chunk_tokens=400)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    assert all(c.role == ChunkRole.CHILD for c in chunks)
    assert all(c.parent_content for c in chunks)
    assert all(c.chunk_id for c in chunks)
