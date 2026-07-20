from __future__ import annotations

from collections.abc import Sequence

from src.modules.retrieval.app.ports import IRetrievalStrategy
from src.modules.retrieval.domain.plan import RetrievalPlan


class RetrievalStrategyRegistry:
    def __init__(self, strategies: Sequence[IRetrievalStrategy]) -> None:
        self._strategies = tuple(strategies)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(strategy.name for strategy in self._strategies)

    def select(self, plan: RetrievalPlan) -> tuple[IRetrievalStrategy, ...]:
        return tuple(
            strategy for strategy in self._strategies if strategy.supports(plan)
        )
