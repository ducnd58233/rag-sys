from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class HitChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Mapping[str, str]
