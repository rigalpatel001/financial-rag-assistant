from dataclasses import dataclass, field
from pathlib import Path
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
