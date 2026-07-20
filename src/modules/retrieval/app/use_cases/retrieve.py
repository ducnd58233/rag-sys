from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.modules.retrieval.app.dto import (
    RetrievalFilter,
    RetrievedItem,
    RetrieveRequest,
    RetrieveResult,
)
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
from src.shared.observability.metrics import (
    retrieval_router_decisions,
    retrieval_strategy_duration,
    retrieval_strategy_results,
)

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


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
        with _tracer.start_as_current_span("retrieval.retrieve") as span:
            try:
                span.set_attribute("rag.query.length", len(query))
                span.set_attribute("rag.retrieval.top_k", top_k)
                plan = await self._query_router.route(
                    query,
                    filters=request.filters,
                    top_k=top_k,
                )
                span.set_attribute("rag.router.kind", plan.router_kind.value)
                span.set_attribute("rag.router.confidence", plan.confidence)
                span.set_attribute(
                    "rag.retrieval.strategies",
                    tuple(strategy.name for strategy in plan.strategies),
                )
                retrieval_router_decisions.add(
                    1,
                    {
                        "router.kind": plan.router_kind.value,
                        "strategy": _strategy_label(plan),
                    },
                )

                strategies = self._strategy_registry.select(plan)
                if not strategies:
                    raise RetrievalInternalError(
                        "no retrieval strategy supports the plan"
                    )

                ranked_lists = await asyncio.gather(
                    *(
                        _retrieve_with_metrics(
                            strategy,
                            query=query,
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
                span.set_attribute("rag.retrieval.results", len(accepted))
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, error.__class__.__name__))
                raise

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


async def _retrieve_with_metrics(
    strategy: IRetrievalStrategy,
    *,
    query: str,
    plan: RetrievalPlan,
    filters: RetrievalFilter,
) -> Sequence[HitChunk]:
    started_at = time.perf_counter()
    outcome = "success"
    attributes = {
        "strategy": strategy.name,
        "router.kind": plan.router_kind.value,
    }
    with _tracer.start_as_current_span(
        f"retrieval.strategy.{strategy.name}",
        attributes={
            "rag.router.kind": plan.router_kind.value,
            "rag.router.confidence": plan.confidence,
            "rag.retrieval.strategy": strategy.name,
        },
    ) as span:
        try:
            hits = await strategy.retrieve(
                query,
                plan=plan,
                filters=filters,
            )
            span.set_attribute("rag.retrieval.results", len(hits))
            return hits
        except Exception as error:
            outcome = "failure"
            span.set_status(Status(StatusCode.ERROR, error.__class__.__name__))
            raise
        finally:
            duration = time.perf_counter() - started_at
            span.set_attribute("rag.retrieval.duration", duration)
            metric_attributes = {**attributes, "outcome": outcome}
            retrieval_strategy_duration.record(
                duration,
                metric_attributes,
            )
            if outcome == "success":
                retrieval_strategy_results.record(len(hits), metric_attributes)


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


def _strategy_label(plan: RetrievalPlan) -> str:
    return "+".join(strategy.name for strategy in plan.strategies)
