from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RouterKind(StrEnum):
    RULE = "rule"
    LLM = "llm"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class StrategySelection:
    name: str
    weight: float
    top_k: int
    query: str | None = None
    as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "weight": self.weight,
            "top_k": self.top_k,
        }
        if self.query is not None:
            result["query"] = self.query
        if self.as_of is not None:
            result["as_of"] = self.as_of
        return result


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    strategies: tuple[StrategySelection, ...]
    router_kind: RouterKind
    reason: str
    confidence: float

    def selection_for(self, name: str) -> StrategySelection | None:
        return next(
            (strategy for strategy in self.strategies if strategy.name == name),
            None,
        )

    @classmethod
    def single(
        cls,
        *,
        strategy: str,
        top_k: int,
        router_kind: RouterKind,
        reason: str,
        confidence: float = 1.0,
    ) -> RetrievalPlan:
        return cls(
            strategies=(StrategySelection(name=strategy, weight=1.0, top_k=top_k),),
            router_kind=router_kind,
            reason=reason,
            confidence=confidence,
        )

    @classmethod
    def hybrid(
        cls, *, top_k: int, reason: str, confidence: float = 1.0
    ) -> RetrievalPlan:
        return cls.single(
            strategy="hybrid",
            top_k=top_k,
            router_kind=RouterKind.FALLBACK,
            reason=reason,
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "router_kind": self.router_kind.value,
            "reason": self.reason,
            "confidence": self.confidence,
        }
