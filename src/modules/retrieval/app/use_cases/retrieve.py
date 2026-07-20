from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievedItem, RetrieveRequest, RetrieveResult
from src.modules.retrieval.app.ports import (
    IQueryRouter,
    IRankFusion,
    IRetrievalStrategy,
)
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
        query_router: IQueryRouter,
        strategy_registry: RetrievalStrategyRegistry,
        rank_fusion: IRankFusion,
    ) -> None:
        self._retrieval_settings = retrieval_settings
        self._query_router = query_router
        self._strategy_registry = strategy_registry
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
        plan = await self._query_router.route(
            query,
            filters=request.filters,
            top_k=top_k,
        )

        strategies = self._strategy_registry.select(plan)
        if not strategies:
            raise RetrievalInternalError("no retrieval strategy supports the plan")

        ranked_lists = await asyncio.gather(
            *(
                strategy.retrieve(
                    query,
                    plan=plan,
                    filters=request.filters,
                )
                for strategy in strategies
            )
        )
        ranked = (
            tuple(ranked_lists[0])
            if len(ranked_lists) == 1
            else self._rank_fusion.fuse(
                ranked_lists,
                top_k=top_k,
                weights=_strategy_weights(plan, strategies),
            )
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


def _strategy_weights(
    plan: RetrievalPlan,
    strategies: Sequence[IRetrievalStrategy],
) -> tuple[float, ...]:
    weights: list[float] = []
    for strategy in strategies:
        selection = plan.selection_for(strategy.name)
        if selection is not None:
            weights.append(selection.weight)
    return tuple(weights)
