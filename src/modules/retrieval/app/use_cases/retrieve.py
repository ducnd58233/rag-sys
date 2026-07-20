from __future__ import annotations

import logging
from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievedItem, RetrieveRequest, RetrieveResult
from src.modules.retrieval.app.strategy_registry import RetrievalStrategyRegistry
from src.modules.retrieval.domain.errors import (
    RetrievalInternalError,
    RetrievalValidationError,
)
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan
from src.shared.configs.settings import RetrievalSettings

logger = logging.getLogger(__name__)


class RetrieveUseCase:
    def __init__(
        self,
        retrieval_settings: RetrievalSettings,
        strategy_registry: RetrievalStrategyRegistry,
    ) -> None:
        self._retrieval_settings = retrieval_settings
        self._strategy_registry = strategy_registry

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

        strategies = self._strategy_registry.select(plan)
        if not strategies:
            raise RetrievalInternalError("no retrieval strategy supports the plan")
        ranked = await strategies[0].retrieve(
            query,
            plan=plan,
            filters=request.filters,
        )
        accepted = self._apply_fused_score_gate(ranked)

        logger.info(
            "retrieve query_len=%d strategy=%s fused=%d accepted=%d top_k=%d min_fused_score=%s",
            len(query),
            ",".join(strategy.name for strategy in strategies),
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
