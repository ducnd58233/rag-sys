from __future__ import annotations
import asyncio
import logging

from src.modules.retrieval.app.ports import IDenseSearcher, ILexicalSearcher, IRankFusion
from src.modules.retrieval.domain.errors import RetrievalValidationError
from src.modules.retrieval.domain.models import HitChunk
from src.shared.app.ports import IEmbeddingModel
from src.modules.retrieval.app.dto import RetrieveRequest, RetrieveResult, RetrievedItem
from src.shared.configs.settings import RetrievalSettings

logger = logging.getLogger(__name__)

class RetrieveUseCase:
    def __init__(
        self,
        retrieval_settings: RetrievalSettings,
        embedder: IEmbeddingModel,
        lexical_searcher: ILexicalSearcher,
        dense_searcher: IDenseSearcher,
        rank_fusion: IRankFusion,
    ) -> None:
        self._retrieval_settings = retrieval_settings
        self._embedder = embedder
        self._lexical_searcher = lexical_searcher
        self._dense_searcher = dense_searcher
        self._rank_fusion = rank_fusion

    async def execute(self, request: RetrieveRequest) -> RetrieveResult:
        query = request.query.strip()
        if not query:
            raise RetrievalValidationError("query cannot be empty")
        
        top_k = request.top_k if request.top_k is not None else self._retrieval_settings.top_k
        if top_k < 1:
            raise RetrievalValidationError("top_k must be greater than 0")

        vectors = await self._embedder.embed([query])
        if not vectors or not vectors[0]:
            raise RetrievalValidationError("failed to embed query")
        
        query_vector = vectors[0]

        lexical_hits, dense_hits = await asyncio.gather(
            self._lexical_searcher.search(
                query, 
                limit=self._retrieval_settings.num_candidates, 
                document_id=request.document_id,
            ),
            self._dense_searcher.search(
                query_vector, 
                limit=self._retrieval_settings.num_candidates, 
                num_candidates=self._retrieval_settings.candidate_k, 
                document_id=request.document_id,
            ),
        )

        ranked = self._rank_fusion.fuse(
            [lexical_hits, dense_hits],
            top_k=top_k,
        )
        accepted = self._apply_fused_score_gate(ranked)

        logger.info(
            "retrieve query_len=%d lexical=%d dense=%d fused=%d accepted=%d top_k=%d min_fused_score=%s",
            len(query),
            len(lexical_hits),
            len(dense_hits),
            len(ranked),
            len(accepted),
            top_k,
            self._retrieval_settings.min_fused_score,
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
                for hit in accepted
            ],
        )
    
    def _apply_fused_score_gate(
        self,
        ranked: list[HitChunk],
    ) -> tuple[HitChunk, ...]:
        threshold = self._retrieval_settings.min_fused_score
        if threshold is None:
            return ranked
        return tuple(hit for hit in ranked if hit.score >= threshold)