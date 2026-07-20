from __future__ import annotations

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import IQueryRouter
from src.modules.retrieval.app.router.rules import RuleRouter
from src.modules.retrieval.domain.plan import RetrievalPlan


class CompositeQueryRouter(IQueryRouter):
    def __init__(
        self,
        rule_router: RuleRouter,
        llm_router: IQueryRouter,
    ) -> None:
        self._rule_router = rule_router
        self._llm_router = llm_router

    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan:
        try:
            rule_plan = await self._rule_router.route(
                query,
                filters=filters,
                top_k=top_k,
            )
        except Exception:
            return RetrievalPlan.hybrid(top_k=top_k, reason="router_error")
        if rule_plan is not None:
            return rule_plan
        try:
            return await self._llm_router.route(
                query,
                filters=filters,
                top_k=top_k,
            )
        except Exception:
            return RetrievalPlan.hybrid(top_k=top_k, reason="llm_router_error")
