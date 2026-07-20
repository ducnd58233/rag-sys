from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.modules.retrieval.domain.plan import RetrievalPlan


@dataclass(frozen=True, slots=True)
class RetrievalFilter:
    org_id: int
    document_id: int | None = None
    document_version_id: int | None = None
    as_of: datetime | None = None


@dataclass(frozen=True, slots=True)
class RetrieveRequest:
    query: str
    filters: RetrievalFilter
    top_k: int | None = None


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
    plan: RetrievalPlan
