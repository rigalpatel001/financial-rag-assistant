from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    page_number: int
    text: str


@dataclass
class FinancialDocument:
    document_id: str
    filename: str
    pages: list[DocumentPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
