"""
Parent-child chunking.

• Parent blocks (~1500 tokens): full section context stored in payload.
• Child chunks (~300 tokens): embedded and retrieved; each carries its parent text.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterator

from cortex.ingestion.chunking.token_counter import count_tokens, truncate_to_tokens
from cortex.models.document import Chunk, Document, Section
from cortex.models.enums import ChunkRole

log = logging.getLogger(__name__)


class ParentChildChunker:
    def __init__(
        self,
        child_chunk_tokens: int = 300,
        parent_chunk_tokens: int = 1500,
        chunk_overlap_tokens: int = 50,
    ) -> None:
        self.child_chunk_tokens = child_chunk_tokens
        self.parent_chunk_tokens = parent_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        heading_stack: list[tuple[int, str]] = []

        for parent_block, heading_path in self._iter_parent_blocks(document, heading_stack):
            parent_id = self._chunk_id(document.doc_id, chunk_index)
            child_texts = list(
                self._split_with_overlap(
                    parent_block,
                    self.child_chunk_tokens,
                    self.chunk_overlap_tokens,
                )
            )

            if not child_texts:
                continue

            for child_text in child_texts:
                chunk_id = self._chunk_id(document.doc_id, chunk_index)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        role=ChunkRole.CHILD,
                        content=child_text,
                        parent_content=parent_block,
                        heading_path=heading_path,
                        source_type=document.source_type,
                        metadata={
                            "parent_chunk_id": parent_id,
                            **document.metadata,
                            "title": document.title,
                            "relative_path": document.relative_path,
                            "file_hash": document.file_hash,
                        },
                    )
                )
                chunk_index += 1

        log.debug(
            "document_chunked",
            extra={"doc_id": document.doc_id, "chunk_count": len(chunks)},
        )
        return chunks

    def _iter_parent_blocks(
        self,
        document: Document,
        heading_stack: list[tuple[int, str]],
    ) -> Iterator[tuple[str, str]]:
        buffer_parts: list[str] = []
        buffer_tokens = 0
        buffer_heading_path = ""

        for section in document.sections:
            heading_path = self._update_heading_stack(heading_stack, section)
            section_text = self._format_section(section, heading_path)
            section_tokens = count_tokens(section_text)

            if section_tokens > self.parent_chunk_tokens:
                if buffer_parts:
                    yield "\n\n".join(buffer_parts), buffer_heading_path
                    buffer_parts = []
                    buffer_tokens = 0

                for piece in self._split_with_overlap(
                    section_text,
                    self.parent_chunk_tokens,
                    overlap_tokens=0,
                ):
                    yield piece, heading_path
                continue

            if buffer_tokens + section_tokens > self.parent_chunk_tokens and buffer_parts:
                yield "\n\n".join(buffer_parts), buffer_heading_path
                buffer_parts = [section_text]
                buffer_tokens = section_tokens
                buffer_heading_path = heading_path
            else:
                buffer_parts.append(section_text)
                buffer_tokens += section_tokens
                if not buffer_heading_path:
                    buffer_heading_path = heading_path

        if buffer_parts:
            yield "\n\n".join(buffer_parts), buffer_heading_path

    def _update_heading_stack(
        self,
        stack: list[tuple[int, str]],
        section: Section,
    ) -> str:
        if section.heading_level == 0:
            return " > ".join(text for _, text in stack) if stack else ""

        level = section.heading_level
        while stack and stack[-1][0] >= level:
            stack.pop()
        if section.heading_text:
            stack.append((level, section.heading_text))
        return " > ".join(text for _, text in stack)

    def _format_section(self, section: Section, heading_path: str) -> str:
        if section.heading_level == 0:
            return section.content
        prefix = section.heading_text or heading_path
        return f"{prefix}\n\n{section.content}"

    def _split_with_overlap(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> Iterator[str]:
        if count_tokens(text) <= max_tokens:
            yield text.strip()
            return

        words = text.split()
        if not words:
            return

        start = 0
        while start < len(words):
            chunk_words: list[str] = []
            token_count = 0
            idx = start

            while idx < len(words) and token_count < max_tokens:
                candidate = " ".join(chunk_words + [words[idx]])
                candidate_tokens = count_tokens(candidate)
                if candidate_tokens > max_tokens and chunk_words:
                    break
                chunk_words.append(words[idx])
                token_count = candidate_tokens
                idx += 1

            if not chunk_words:
                chunk_words = [truncate_to_tokens(words[start], max_tokens)]
                idx = start + 1

            yield " ".join(chunk_words).strip()

            if idx >= len(words):
                break

            overlap_count = 0
            overlap_start = max(start, idx - 1)
            while overlap_start > start and overlap_count < overlap_tokens:
                overlap_start -= 1
                overlap_count = count_tokens(" ".join(words[overlap_start:idx]))

            start = overlap_start if overlap_start > start else idx

    @staticmethod
    def _chunk_id(doc_id: str, chunk_index: int) -> str:
        raw = f"{doc_id}:{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()
