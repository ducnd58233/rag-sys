from dataclasses import FrozenInstanceError

import pytest

from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)


def test_retrieval_plan_serializes_to_plain_values() -> None:
    plan = RetrievalPlan(
        strategies=(
            StrategySelection(name="lexical", weight=0.7, top_k=8),
            StrategySelection(name="semantic", weight=0.3, top_k=8),
        ),
        router_kind=RouterKind.RULE,
        reason="identifier",
        confidence=0.95,
    )

    assert plan.to_dict() == {
        "strategies": [
            {"name": "lexical", "weight": 0.7, "top_k": 8},
            {"name": "semantic", "weight": 0.3, "top_k": 8},
        ],
        "router_kind": "rule",
        "reason": "identifier",
        "confidence": 0.95,
    }


def test_retrieval_plan_is_immutable() -> None:
    plan = RetrievalPlan.hybrid(top_k=5, reason="fallback")

    with pytest.raises(FrozenInstanceError):
        plan.reason = "changed"
