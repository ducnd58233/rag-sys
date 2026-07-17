from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RetrieveRequest:
    query: str
    top_k: int | None = None
    document_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RetrieveResult:
    query: str
    items: Sequence[RetrievedItem]