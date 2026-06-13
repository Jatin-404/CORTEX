"""Parser protocol — every source format implements this interface."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from cortex.models.document import Document


class DocumentParser(Protocol):
    """Yield unified Document objects from a source corpus."""

    def iter_documents(self) -> Iterator[Document]:
        ...

    def parse_file(self, path: Path) -> Document | None:
        ...
