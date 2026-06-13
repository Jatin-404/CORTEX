"""
Handbook-aware markdown parser for GitLab-style handbook corpora.

Expected layout:
  content/handbook/<department>/<optional-sub>/<file>.md
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Iterator

import frontmatter

from cortex.models.document import Document, Section
from cortex.models.enums import SourceType

log = logging.getLogger(__name__)

_NS = uuid.UUID("c5b9f94e-3e18-4e7d-b8a1-2f09dcb1e823")

_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://\S+")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


class HandbookParser:
    """Parse GitLab handbook markdown into unified Document objects."""

    def __init__(self, handbook_root: Path) -> None:
        self.root = handbook_root.resolve()

    def iter_documents(self) -> Iterator[Document]:
        md_files = sorted(self.root.rglob("*.md"))
        log.info("handbook_scan_complete", extra={"file_count": len(md_files)})
        for path in md_files:
            doc = self.parse_file(path)
            if doc is not None:
                yield doc

    def parse_file(self, path: Path) -> Document | None:
        path = path.resolve()
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("handbook_read_failed", extra={"path": str(path), "error": str(exc)})
            return None

        try:
            post = frontmatter.loads(raw_text)
        except Exception as exc:
            log.warning("handbook_frontmatter_failed", extra={"path": str(path), "error": str(exc)})
            return None

        rel_path = self._relative_path(path)
        department, subdepartment = self._extract_taxonomy(path)

        title = str(
            post.metadata.get("title")
            or post.metadata.get("name")
            or path.stem.replace("-", " ").title()
        )

        sections = self._split_sections(post.content)
        if not sections and not post.content.strip():
            log.debug("handbook_empty_document", extra={"path": rel_path})
            return None

        return Document(
            doc_id=self._doc_id(rel_path),
            title=title,
            source_type=SourceType.HANDBOOK_MARKDOWN,
            source_path=path,
            relative_path=rel_path,
            sections=sections,
            file_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            raw_text=raw_text,
            metadata={
                "department": department,
                "subdepartment": subdepartment,
                "confidentiality_level": str(
                    post.metadata.get("confidentiality_level", "global")
                ),
                "last_updated": str(post.metadata.get("last_updated", "")),
                "is_index": path.name == "_index.md",
                "frontmatter": dict(post.metadata),
            },
        )

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def _doc_id(self, rel_path: str) -> str:
        return str(uuid.uuid5(_NS, rel_path))

    def _extract_taxonomy(self, path: Path) -> tuple[str, str]:
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            return ("general", "")

        if len(parts) == 1:
            return ("handbook", "")

        department = parts[0]
        subdepartment = parts[1] if len(parts) >= 3 else ""
        return department, subdepartment

    def _split_sections(self, content: str) -> list[Section]:
        heading_matches = list(_HEADING_RE.finditer(content))
        boundaries: list[tuple[int, str, int, int]] = []

        for i, match in enumerate(heading_matches):
            end = (
                heading_matches[i + 1].start()
                if i + 1 < len(heading_matches)
                else len(content)
            )
            boundaries.append(
                (match.start(), match.group(2).strip(), len(match.group(1)), end)
            )

        sections: list[Section] = []

        preamble_end = boundaries[0][0] if boundaries else len(content)
        preamble_text = self._clean(content[:preamble_end]).strip()
        if preamble_text:
            sections.append(
                Section(heading_level=0, heading_text="", content=preamble_text, char_start=0)
            )

        for start_pos, heading_text, level, end_pos in boundaries:
            section_body = content[start_pos:end_pos]
            first_newline = section_body.find("\n")
            body_text = section_body[first_newline + 1 :] if first_newline != -1 else ""
            cleaned = self._clean(body_text).strip()
            if cleaned:
                sections.append(
                    Section(
                        heading_level=level,
                        heading_text=heading_text,
                        content=cleaned,
                        char_start=start_pos,
                    )
                )

        return sections

    @staticmethod
    def _clean(text: str) -> str:
        text = _LINK_RE.sub(r"\1", text)
        text = _URL_RE.sub("", text)
        return text
