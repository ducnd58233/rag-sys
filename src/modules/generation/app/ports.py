from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.modules.generation.domain.models import ContextChunk

class IContextRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[ContextChunk]: ...