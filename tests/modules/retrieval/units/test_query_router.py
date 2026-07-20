import pytest

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.router import CompositeQueryRouter, RuleRouter
from src.modules.retrieval.domain.plan import RouterKind
from src.shared.configs.settings import RoutingSettings


@pytest.mark.asyncio
async def test_rule_router_routes_explicit_filters_to_structured() -> None:
    router = RuleRouter(_routing_settings())

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
async def test_rule_router_routes_identifiers_from_configured_patterns() -> None:
    router = RuleRouter(_routing_settings(identifier_patterns=[r"deploy-\d+"]))

    plan = await router.route(
        "show deploy-1832 status",
        filters=RetrievalFilter(org_id=1),
        top_k=7,
    )

    assert plan is not None
    assert [(strategy.name, strategy.query) for strategy in plan.strategies] == [
        ("structured", "deploy-1832"),
        ("lexical", None),
    ]


@pytest.mark.asyncio
async def test_rule_router_orders_explicit_filter_before_identifier() -> None:
    router = RuleRouter(_routing_settings(identifier_patterns=[r"deploy-\d+"]))

    plan = await router.route(
        "show deploy-1832",
        filters=RetrievalFilter(org_id=1, document_id=42),
        top_k=3,
    )

    assert plan is not None
    assert [strategy.name for strategy in plan.strategies] == ["structured"]
    assert plan.reason == "explicit_filter"


@pytest.mark.asyncio
async def test_rule_router_leaves_temporal_and_relationship_queries_for_llm() -> None:
    temporal_router = RuleRouter(_routing_settings())
    relationship_router = RuleRouter(_routing_settings())

    temporal = await temporal_router.route(
        "policy as of 2024",
        filters=RetrievalFilter(org_id=1),
        top_k=4,
    )
    relationship = await relationship_router.route(
        "what depends on checkout",
        filters=RetrievalFilter(org_id=1),
        top_k=4,
    )

    assert temporal is None
    assert relationship is None


@pytest.mark.asyncio
async def test_rule_router_falls_through_without_rule_match() -> None:
    router = RuleRouter(_routing_settings(identifier_patterns=[r"deploy-\d+"]))

    plan = await router.route(
        "why did checkout latency regress",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )

    assert plan is None


@pytest.mark.asyncio
async def test_composite_router_caps_rule_strategies_and_uses_default_fallback() -> (
    None
):
    settings = _routing_settings(
        max_concurrent_strategies=1, default_strategy="lexical"
    )
    router = CompositeQueryRouter(settings, RuleRouter(settings))

    rule_plan = await router.route(
        "deploy-1832",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )
    fallback_plan = await router.route(
        "plain semantic query",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )

    assert [strategy.name for strategy in rule_plan.strategies] == ["structured"]
    assert [strategy.name for strategy in fallback_plan.strategies] == ["lexical"]
    assert fallback_plan.router_kind == RouterKind.FALLBACK
    assert fallback_plan.reason == "llm_router_disabled"


@pytest.mark.asyncio
async def test_composite_router_falls_back_when_rule_router_errors() -> None:
    router = CompositeQueryRouter(_routing_settings(), BrokenRuleRouter())

    plan = await router.route(
        "deploy-1832",
        filters=RetrievalFilter(org_id=1),
        top_k=5,
    )

    assert [strategy.name for strategy in plan.strategies] == ["hybrid"]
    assert plan.reason == "router_error"


class BrokenRuleRouter:
    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ):
        raise RuntimeError("boom")


def _routing_settings(**overrides: object) -> RoutingSettings:
    values = {
        "identifier_patterns": [r"[a-z]+-\d{3,}"],
        **overrides,
    }
    return RoutingSettings(**values)
