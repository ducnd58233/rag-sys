from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping

@dataclass(frozen=True, slots=True)
class ContextChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document_id: str
    
@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    text: str
    citations: tuple[Citation, ...]
    refused: bool