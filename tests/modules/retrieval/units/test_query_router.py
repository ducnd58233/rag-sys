import pytest

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.router import CompositeQueryRouter, RuleRouter
from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)


@pytest.mark.asyncio
async def test_rule_router_routes_explicit_filters_to_structured() -> None:
    router = RuleRouter()

    plan = await router.route(
        "deploy-1832",
        filters=RetrievalFilter(org_id=1, document_id=42),
        top_k=5,
    )

    assert plan is not None
    assert [strategy.name for strategy in plan.strategies] == ["structured"]
    assert plan.reason == "explicit_filter"
    assert plan.router_kind == RouterKind.RULE


@pytest.mark.asyncio
async def test_rule_router_leaves_everything_else_for_the_llm() -> None:
    router = RuleRouter()

    for query in (
        "show deploy-1832 status",
        "policy as of 2024",
        "what depends on checkout",
        "why did checkout latency regress",
    ):
        plan = await router.route(query, filters=RetrievalFilter(org_id=1), top_k=4)
        assert plan is None


@pytest.mark.asyncio
async def test_composite_router_routes_unmatched_queries_to_llm() -> None:
    llm_router = FakeLlmRouter(
        RetrievalPlan.single(
            strategy="semantic",
            top_k=5,
            router_kind=RouterKind.LLM,
            reason="semantic",
        )
    )
    router = CompositeQueryRouter(RuleRouter(), llm_router)

    rule_plan = await router.route(
        "deploy-1832",
        filters=RetrievalFilter(org_id=1, document_id=42),
        top_k=5,
    )
    llm_plan = await router.route(
        "plain semantic query",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )

    assert [strategy.name for strategy in rule_plan.strategies] == ["structured"]
    assert [strategy.name for strategy in llm_plan.strategies] == ["semantic"]
    assert llm_plan.router_kind == RouterKind.LLM
    assert llm_router.calls == [("plain semantic query", RetrievalFilter(org_id=1), 5)]


@pytest.mark.asyncio
async def test_composite_router_falls_back_when_rule_router_errors() -> None:
    router = CompositeQueryRouter(
        BrokenRuleRouter(),
        FakeLlmRouter(
            RetrievalPlan(
                strategies=(StrategySelection(name="semantic", weight=1.0, top_k=5),),
                router_kind=RouterKind.LLM,
                reason="semantic",
                confidence=1.0,
            )
        ),
    )

    plan = await router.route(
        "deploy-1832",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )

    assert [strategy.name for strategy in plan.strategies] == ["hybrid"]
    assert plan.reason == "router_error"


@pytest.mark.asyncio
async def test_composite_router_falls_back_when_llm_router_errors() -> None:
    router = CompositeQueryRouter(RuleRouter(), BrokenLlmRouter())

    plan = await router.route(
        "plain query",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )

    assert [strategy.name for strategy in plan.strategies] == ["hybrid"]
    assert plan.reason == "llm_router_error"


class BrokenRuleRouter:
    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ):
        raise RuntimeError("boom")


class BrokenLlmRouter:
    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ):
        raise RuntimeError("boom")


class FakeLlmRouter:
    def __init__(self, plan: RetrievalPlan) -> None:
        self.calls: list[tuple[str, RetrievalFilter, int]] = []
        self._plan = plan

    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan:
        self.calls.append((query, filters, top_k))
        return self._plan
