import asyncio
from collections.abc import Sequence

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.router import CompositeQueryRouter, LlmRouter, RuleRouter
from src.modules.retrieval.app.router.llm_schema import RouterPlanSchema
from src.modules.retrieval.domain.plan import RouterKind
from src.shared.configs.settings import RoutingSettings


class FakeChatModel:
    def __init__(self, results: Sequence[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._results = list(results)

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[RouterPlanSchema],
        temperature: float | None = None,
    ) -> RouterPlanSchema:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
            }
        )
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return schema.model_validate(result)

    async def complete(self, **kwargs: object):
        raise AssertionError("complete should not be called")

    def bind_tools(self, tools: Sequence[object]):
        return self


class SlowChatModel(FakeChatModel):
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[RouterPlanSchema],
        temperature: float | None = None,
    ) -> RouterPlanSchema:
        await asyncio.sleep(0.05)
        return await super().complete_structured(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
        )


@pytest.mark.asyncio
async def test_llm_router_uses_structured_temperature_zero_and_keeps_all_selected() -> (
    None
):
    chat = FakeChatModel(
        [
            {
                "strategies": [
                    {"name": "semantic", "weight": 0.2},
                    {"name": "lexical", "weight": 0.9, "query": "policy version"},
                    {"name": "structured", "weight": 0.5},
                ],
                "reason": "mixed",
                "confidence": 0.7,
            }
        ]
    )
    router = LlmRouter(
        RoutingSettings(),
        chat,
        allowed_strategies=("lexical", "semantic", "structured"),
    )

    plan = await router.route("compare policy versions", filters=_filters(), top_k=6)

    # No cap: every selected strategy comes through, sorted by weight.
    assert [strategy.name for strategy in plan.strategies] == [
        "lexical",
        "structured",
        "semantic",
    ]
    assert [strategy.query for strategy in plan.strategies] == [
        "policy version",
        None,
        None,
    ]
    assert [strategy.top_k for strategy in plan.strategies] == [6, 6, 6]
    assert plan.router_kind == RouterKind.LLM
    assert plan.reason == "mixed"
    assert chat.calls[0]["temperature"] == 0.0
    assert "max_tokens" not in chat.calls[0]


@pytest.mark.asyncio
async def test_llm_router_carries_temporal_and_graph_parameters() -> None:
    chat = FakeChatModel(
        [
            {
                "strategies": [
                    {
                        "name": "temporal",
                        "weight": 0.7,
                        "query": "checkout policy",
                        "as_of": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "name": "graph",
                        "weight": 0.3,
                        "query": "checkout dependencies",
                    },
                ],
                "reason": "temporal relationship",
                "confidence": 0.8,
            }
        ]
    )
    router = LlmRouter(
        RoutingSettings(),
        chat,
        allowed_strategies=("temporal", "graph", "hybrid"),
    )

    plan = await router.route(
        "what depended on checkout as of 2026-01-01",
        filters=_filters(),
        top_k=5,
    )

    assert [(item.name, item.query, item.as_of) for item in plan.strategies] == [
        ("temporal", "checkout policy", "2026-01-01T00:00:00+00:00"),
        ("graph", "checkout dependencies", None),
    ]


@pytest.mark.asyncio
async def test_llm_router_drops_unknown_names_and_falls_back_when_empty() -> None:
    router = LlmRouter(
        RoutingSettings(),
        FakeChatModel(
            [
                {
                    "strategies": [{"name": "unknown", "weight": 1.0}],
                    "reason": "bad",
                    "confidence": 0.8,
                },
                {
                    "strategies": [{"name": "other", "weight": 1.0}],
                    "reason": "bad again",
                    "confidence": 0.8,
                },
            ]
        ),
        allowed_strategies=("lexical",),
    )

    plan = await router.route("query", filters=_filters(), top_k=4)

    assert [strategy.name for strategy in plan.strategies] == ["hybrid"]
    assert plan.router_kind == RouterKind.FALLBACK
    assert plan.reason == "llm_router_empty"


@pytest.mark.asyncio
async def test_llm_router_retries_malformed_output_then_uses_valid_result() -> None:
    chat = FakeChatModel(
        [
            RuntimeError("malformed"),
            {
                "strategies": [{"name": "semantic", "weight": 1.0}],
                "reason": "semantic",
                "confidence": 0.6,
            },
        ]
    )
    router = LlmRouter(
        RoutingSettings(),
        chat,
        allowed_strategies=("semantic",),
    )

    plan = await router.route("query", filters=_filters(), top_k=3)

    assert [strategy.name for strategy in plan.strategies] == ["semantic"]
    assert len(chat.calls) == 2


@pytest.mark.asyncio
async def test_llm_router_timeout_falls_back_to_hybrid() -> None:
    router = LlmRouter(
        RoutingSettings(llm_router_timeout_seconds=0.001),
        SlowChatModel(
            [
                {
                    "strategies": [{"name": "semantic", "weight": 1.0}],
                    "reason": "semantic",
                    "confidence": 0.6,
                }
            ]
        ),
        allowed_strategies=("semantic",),
    )

    plan = await router.route("query", filters=_filters(), top_k=3)

    assert [strategy.name for strategy in plan.strategies] == ["hybrid"]
    assert plan.reason == "llm_router_timeout"


@pytest.mark.asyncio
async def test_composite_router_uses_llm_router_when_available() -> None:
    llm_router = LlmRouter(
        RoutingSettings(),
        FakeChatModel(
            [
                {
                    "strategies": [{"name": "semantic", "weight": 1.0}],
                    "reason": "semantic",
                    "confidence": 0.6,
                }
            ]
        ),
        allowed_strategies=("semantic",),
    )
    router = CompositeQueryRouter(RuleRouter(), llm_router)

    plan = await router.route("explain architecture", filters=_filters(), top_k=5)

    assert [strategy.name for strategy in plan.strategies] == ["semantic"]
    assert plan.router_kind == RouterKind.LLM


def _filters() -> RetrievalFilter:
    return RetrievalFilter(org_id=1)
