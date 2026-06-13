"""Unified document model — all parsers normalize to this shape."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cortex.models.enums import ChunkRole, SourceType


@dataclass
class Section:
    heading_level: int
    heading_text: str
    content: str
    char_start: int = 0


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: SourceType
    source_path: Path
    relative_path: str
    sections: list[Section]
    file_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class Chunk:
    """A retrievable unit ready for embedding and Qdrant upsert."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    role: ChunkRole
    content: str
    parent_content: str
    heading_path: str
    source_type: SourceType
    metadata: dict[str, Any] = field(default_factory=dict)
