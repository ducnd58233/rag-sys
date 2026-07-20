from __future__ import annotations

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)


class RuleRouter:
    """Honors an explicit document scope - not an LLM routing decision."""

    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan | None:
        if filters.document_id is None and filters.document_version_id is None:
            return None
        return RetrievalPlan(
            strategies=(StrategySelection(name="structured", weight=1.0, top_k=top_k),),
            router_kind=RouterKind.RULE,
            reason="explicit_filter",
            confidence=1.0,
        )
