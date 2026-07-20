from __future__ import annotations

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import IQueryRouter
from src.modules.retrieval.app.router.rules import RuleRouter
from src.modules.retrieval.domain.plan import RetrievalPlan, RouterKind
from src.shared.configs.settings import RoutingSettings


class CompositeQueryRouter(IQueryRouter):
    def __init__(
        self,
        settings: RoutingSettings,
        rule_router: RuleRouter,
        llm_router: IQueryRouter,
    ) -> None:
        self._settings = settings
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
            return self._fallback_plan(top_k, reason="router_error")
        if rule_plan is not None:
            return _cap_plan(rule_plan, self._settings.max_concurrent_strategies)
        try:
            return _cap_plan(
                await self._llm_router.route(
                    query,
                    filters=filters,
                    top_k=top_k,
                ),
                self._settings.max_concurrent_strategies,
            )
        except Exception:
            return self._fallback_plan(top_k, reason="llm_router_error")

    def _fallback_plan(self, top_k: int, *, reason: str) -> RetrievalPlan:
        return RetrievalPlan.single(
            strategy=self._settings.default_strategy,
            top_k=top_k,
            router_kind=RouterKind.FALLBACK,
            reason=reason,
        )


def _cap_plan(plan: RetrievalPlan, max_strategies: int) -> RetrievalPlan:
    strategies = tuple(
        sorted(plan.strategies, key=lambda strategy: strategy.weight, reverse=True)[
            :max_strategies
        ]
    )
    return RetrievalPlan(
        strategies=strategies,
        router_kind=plan.router_kind,
        reason=plan.reason,
        confidence=plan.confidence,
    )
