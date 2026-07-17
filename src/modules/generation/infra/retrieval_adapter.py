from __future__ import annotations

from collections.abc import Sequence
from src.modules.generation.domain.models import ContextChunk
from src.modules.retrieval import RetrieveUseCase
from src.modules.retrieval.app.dto import RetrieveRequest


class RetrieveUseCaseAdapter:
    def __init__(self, retrieve: RetrieveUseCase) -> None:
        self._retrieve = retrieve

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[ContextChunk]:
        result = await self._retrieve.execute(
            RetrieveRequest(query=query, top_k=top_k, document_id=document_id)
        )
        return tuple(
            ContextChunk(
                chunk_id=i.chunk_id,
                document_id=i.document_id,
                content=i.content,
                score=i.score,
                metadata=i.metadata,
            )
            for i in result.items
        )