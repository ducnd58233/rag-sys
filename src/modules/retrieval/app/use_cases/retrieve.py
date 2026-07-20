from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievedItem, RetrieveRequest, RetrieveResult
from src.modules.retrieval.app.ports import (
    IDenseSearcher,
    ILexicalSearcher,
    IRankFusion,
)
from src.modules.retrieval.domain.errors import RetrievalValidationError
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan
from src.shared.app.ports import IEmbeddingModel
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

        if request.filters.org_id <= 0:
            raise RetrievalValidationError(
                "org_id must be greater than 0",
            )

        top_k = (
            request.top_k
            if request.top_k is not None
            else self._retrieval_settings.top_k
        )
        if top_k < 1:
            raise RetrievalValidationError("top_k must be greater than 0")
        plan = RetrievalPlan.hybrid(top_k=top_k, reason="hybrid_default")

        vectors = await self._embedder.embed([query])
        if not vectors or not vectors[0]:
            raise RetrievalValidationError("failed to embed query")

        query_vector = vectors[0]

        lexical_hits, dense_hits = await asyncio.gather(
            self._lexical_searcher.search(
                query,
                limit=self._retrieval_settings.num_candidates,
                filters=request.filters,
            ),
            self._dense_searcher.search(
                query_vector,
                limit=self._retrieval_settings.num_candidates,
                num_candidates=self._retrieval_settings.candidate_k,
                filters=request.filters,
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
            plan=plan,
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
        ranked: Sequence[HitChunk],
    ) -> tuple[HitChunk, ...]:
        threshold = self._retrieval_settings.min_fused_score
        if threshold is None:
            return tuple(ranked)
        return tuple(hit for hit in ranked if hit.score >= threshold)
