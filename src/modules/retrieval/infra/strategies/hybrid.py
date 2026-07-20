from __future__ import annotations

import asyncio
from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import IRankFusion, IRetrievalStrategy
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan, StrategySelection


class HybridStrategy(IRetrievalStrategy):
    def __init__(
        self,
        lexical: IRetrievalStrategy,
        semantic: IRetrievalStrategy,
        rank_fusion: IRankFusion,
    ) -> None:
        self._lexical = lexical
        self._semantic = semantic
        self._rank_fusion = rank_fusion

    @property
    def name(self) -> str:
        return "hybrid"

    def supports(self, plan: RetrievalPlan) -> bool:
        return plan.selection_for(self.name) is not None

    async def retrieve(
        self,
        query: str,
        *,
        plan: RetrievalPlan,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        selection = plan.selection_for(self.name)
        if selection is None:
            return ()
        lexical_plan = RetrievalPlan(
            strategies=(
                plan.selection_for(self._lexical.name)
                or StrategySelection(
                    name=self._lexical.name,
                    weight=selection.weight,
                    top_k=selection.top_k,
                ),
            ),
            router_kind=plan.router_kind,
            reason=plan.reason,
            confidence=plan.confidence,
        )
        semantic_plan = RetrievalPlan(
            strategies=(
                plan.selection_for(self._semantic.name)
                or StrategySelection(
                    name=self._semantic.name,
                    weight=selection.weight,
                    top_k=selection.top_k,
                ),
            ),
            router_kind=plan.router_kind,
            reason=plan.reason,
            confidence=plan.confidence,
        )
        lexical_hits, semantic_hits = await asyncio.gather(
            self._lexical.retrieve(
                query,
                plan=lexical_plan,
                filters=filters,
            ),
            self._semantic.retrieve(
                query,
                plan=semantic_plan,
                filters=filters,
            ),
        )
        return self._rank_fusion.fuse(
            [lexical_hits, semantic_hits],
            top_k=selection.top_k,
        )
