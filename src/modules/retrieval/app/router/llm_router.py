from __future__ import annotations

import asyncio

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.router.llm_schema import RouterPlanSchema
from src.modules.retrieval.domain.plan import RetrievalPlan, RouterKind, StrategySelection
from src.shared.app.ports import IChatModel
from src.shared.configs.settings import RoutingSettings

_SYSTEM = """
You route one RAG retrieval query to the best retrieval strategies.
Return JSON with:
- strategies: list of {name, weight}
- reason: short reason
- confidence: number from 0 to 1

Use only strategy names provided by the user message.
Prefer one strategy for confident cases.
Use multiple strategies when evidence needs different retrieval modes.
Do not answer the query.
""".strip()


class LlmRouter:
    def __init__(
        self,
        settings: RoutingSettings,
        chat_model: IChatModel,
        *,
        allowed_strategies: tuple[str, ...],
    ) -> None:
        self._settings = settings
        self._chat = chat_model
        self._allowed_strategies = frozenset(allowed_strategies)

    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan:
        try:
            result = await self._classify_with_timeout(query, filters=filters)
        except TimeoutError:
            return _fallback(top_k, reason="llm_router_timeout")
        except Exception:
            try:
                result = await self._classify_with_timeout(query, filters=filters)
            except Exception:
                return _fallback(top_k, reason="llm_router_invalid")

        plan = _plan_from_result(
            result,
            top_k=top_k,
            allowed_strategies=self._allowed_strategies,
            max_strategies=self._settings.max_concurrent_strategies,
        )
        if plan is None:
            try:
                result = await self._classify_with_timeout(query, filters=filters)
            except Exception:
                return _fallback(top_k, reason="llm_router_invalid")
            plan = _plan_from_result(
                result,
                top_k=top_k,
                allowed_strategies=self._allowed_strategies,
                max_strategies=self._settings.max_concurrent_strategies,
            )
        return plan or _fallback(top_k, reason="llm_router_empty")

    async def _classify_with_timeout(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
    ) -> RouterPlanSchema:
        return await asyncio.wait_for(
            self._classify(query, filters=filters),
            timeout=self._settings.llm_router_timeout_seconds,
        )

    async def _classify(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
    ) -> RouterPlanSchema:
        return await self._chat.complete_structured(
            system=_SYSTEM,
            user=_user_prompt(
                query,
                filters=filters,
                allowed_strategies=tuple(sorted(self._allowed_strategies)),
            ),
            schema=RouterPlanSchema,
            temperature=0.0,
            max_tokens=256,
        )


def _user_prompt(
    query: str,
    *,
    filters: RetrievalFilter,
    allowed_strategies: tuple[str, ...],
) -> str:
    return (
        f"QUERY: {query}\n"
        f"ALLOWED_STRATEGIES: {', '.join(allowed_strategies)}\n"
        f"HAS_DOCUMENT_FILTER: {filters.document_id is not None or filters.document_version_id is not None}"
    )


def _plan_from_result(
    result: RouterPlanSchema,
    *,
    top_k: int,
    allowed_strategies: frozenset[str],
    max_strategies: int,
) -> RetrievalPlan | None:
    selected = [
        StrategySelection(name=item.name, weight=item.weight, top_k=top_k)
        for item in result.strategies
        if item.name in allowed_strategies
    ]
    selected = sorted(selected, key=lambda item: item.weight, reverse=True)[
        :max_strategies
    ]
    if not selected:
        return None
    return RetrievalPlan(
        strategies=tuple(selected),
        router_kind=RouterKind.LLM,
        reason=result.reason.strip() or "llm_router",
        confidence=result.confidence,
    )


def _fallback(top_k: int, *, reason: str) -> RetrievalPlan:
    return RetrievalPlan.hybrid(top_k=top_k, reason=reason)
