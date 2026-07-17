from __future__ import annotations
import asyncio
import logging

from src.modules.retrieval.app.ports import IDenseSearcher, ILexicalSearcher, IRankFusion
from src.modules.retrieval.domain.errors import RetrievalValidationError
from src.shared.app.ports import IEmbeddingModel
from src.modules.retrieval.app.dto import RetrieveRequest, RetrieveResult, RetrievedItem

logger = logging.getLogger(__name__)

class RetrieveUseCase:
    def __init__(
        self,
        embedder: IEmbeddingModel,
        lexical_searcher: ILexicalSearcher,
        dense_searcher: IDenseSearcher,
        rank_fusion: IRankFusion,
        *,
        num_candidates: int = 10,
        candidate_k: int = 100,
    ) -> None:
        self._embedder = embedder
        self._lexical_searcher = lexical_searcher
        self._dense_searcher = dense_searcher
        self._rank_fusion = rank_fusion
        self._num_candidates = num_candidates
        self._candidate_k = candidate_k

    async def execute(self, request: RetrieveRequest) -> RetrieveResult:
        query = request.query.strip()
        if not query:
            raise RetrievalValidationError("query cannot be empty")
        
        if request.top_k < 1:
            raise RetrievalValidationError("top_k must be greater than 0")

        vectors = await self._embedder.embed([query])
        if not vectors or not vectors[0]:
            raise RetrievalValidationError("failed to embed query")
        
        query_vector = vectors[0]

        lexical_hits, dense_hits = await asyncio.gather(
            self._lexical_searcher.search(query, limit=self._num_candidates, document_id=request.document_id),
            self._dense_searcher.search(query_vector, limit=self._num_candidates, num_candidates=self._candidate_k, document_id=request.document_id),
        )

        ranked = self._rank_fusion.fuse(
            [lexical_hits, dense_hits],
            top_k=request.top_k,
        )

        logger.info(
            "retrieve query_len=%d lexical=%d dense=%d fused=%d top_k=%d",
            len(query),
            len(lexical_hits),
            len(dense_hits),
            len(ranked),
            request.top_k,
        )

        return RetrieveResult(
            query=query,
            items=[
                RetrievedItem(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    content=hit.content,
                    score=hit.score,
                    metadata=hit.metadata,
                )
                for hit in ranked
            ],
        )