from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AskRequest:
    org_id: int
    query: str
    top_k: int | None = None
    document_id: int | None = None
    document_version_id: int | None = None
    as_of: datetime | None = None


@dataclass(frozen=True, slots=True)
class CitationItem:
    chunk_id: str
    document_id: str
    content: str
    score: float


@dataclass(frozen=True, slots=True)
class AskResult:
    query: str
    answer: str
    citations: Sequence[CitationItem]
    refused: bool
