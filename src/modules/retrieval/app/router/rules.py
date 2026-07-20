from __future__ import annotations

import re

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)
from src.shared.configs.settings import RoutingSettings


class RuleRouter:
    def __init__(self, settings: RoutingSettings) -> None:
        self._settings = settings
        self._identifier_patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in settings.identifier_patterns
        )

    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan | None:
        if filters.document_id is not None or filters.document_version_id is not None:
            return _plan(
                StrategySelection(name="structured", weight=1.0, top_k=top_k),
                reason="explicit_filter",
            )
        identifier = self._match_identifier(query)
        if identifier is not None:
            return _plan(
                StrategySelection(
                    name="structured",
                    weight=0.6,
                    top_k=top_k,
                    query=identifier,
                ),
                StrategySelection(name="lexical", weight=0.4, top_k=top_k),
                reason="identifier",
                confidence=0.9,
            )
        return None

    def _match_identifier(self, query: str) -> str | None:
        for pattern in self._identifier_patterns:
            match = pattern.search(query)
            if match is not None:
                return match.group(0)
        return None


def _plan(
    *strategies: StrategySelection,
    reason: str,
    confidence: float = 1.0,
) -> RetrievalPlan:
    return RetrievalPlan(
        strategies=tuple(strategies),
        router_kind=RouterKind.RULE,
        reason=reason,
        confidence=confidence,
    )
