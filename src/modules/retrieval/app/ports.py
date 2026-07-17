from collections.abc import Sequence
from typing import Protocol

from src.modules.retrieval.domain.models import HitChunk

class ILexicalSearcher(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int,
        document_id: str | None = None,
    ) -> Sequence[HitChunk]: ...

class IDenseSearcher(Protocol):
    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
        document_id: str | None = None,
    ) -> Sequence[HitChunk]: ...

class IRankFusion(Protocol):
    def fuse(
        self,
        ranked_lists: Sequence[Sequence[HitChunk]],
        *,
        top_k: int,
    ) -> tuple[HitChunk, ...]: ...

