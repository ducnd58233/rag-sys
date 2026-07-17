from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class HitChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    top_k: int
    candidate_k: int
    document_id: str | None = None
