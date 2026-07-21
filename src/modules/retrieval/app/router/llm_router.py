from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.router.llm_schema import RouterPlanSchema
from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)
from src.shared.app.ports import IChatModel
from src.shared.configs.settings import RoutingSettings

_SYSTEM = """
<responsibility>
You are the retrieval router for a RAG system. Given one user query, decide which
retrieval strategies should run against it and what each should search for. Do not
answer the query yourself.
</responsibility>

<strategies>
- hybrid: general-purpose search (fused keyword + semantic vector search). The default
  choice for ordinary questions about content - use this unless another strategy below
  applies more specifically.
- structured: exact-match lookup by a literal identifier (a chunk id, document id,
  filename, or source path). Use only when the query names a specific identifier. Set
  this strategy's query to the literal identifier text extracted from the query, not a
  paraphrase.
- temporal: recency-weighted search. Use when the query asks for the latest, most
  recent, a version, or a state as of a specific time. When a point in time is implied,
  set as_of to an ISO-8601 timestamp inferred from the query.
- graph: entity-relationship traversal. Use when the query asks about relationships,
  dependencies, causes, references, or multi-hop connections between things.
</strategies>

<decision_process>
1. Decide which strategies genuinely apply. Most queries only need hybrid.
2. Combine strategies when the query's evidence needs more than one retrieval mode at
   once (for example, an identifier plus a relationship). There is no cap on how many -
   use as many as the query genuinely calls for, and no more.
3. For every selected strategy other than structured, set query to the best short
   search phrase for that strategy specifically, not a copy of the raw user query.
</decision_process>

<examples>
<example>
<query>What is our refund policy?</query>
<output>{"strategies": [{"name": "hybrid", "weight": 1.0, "query": "refund policy"}], "reason": "general content question", "confidence": 0.9}</output>
</example>
<example>
<query>find chunk_id bioasq-passage-1348618</query>
<output>{"strategies": [{"name": "structured", "weight": 1.0, "query": "bioasq-passage-1348618"}], "reason": "explicit identifier", "confidence": 0.95}</output>
</example>
<example>
<query>what depended on the checkout timeout as of 2026-01-01</query>
<output>{"strategies": [{"name": "temporal", "weight": 0.7, "query": "checkout timeout policy", "as_of": "2026-01-01T00:00:00+00:00"}, {"name": "graph", "weight": 0.3, "query": "checkout timeout dependencies"}], "reason": "temporal relationship", "confidence": 0.8}</output>
</example>
<example>
<query>what does the latest version of document_id 4821 change compared to before</query>
<output>{"strategies": [{"name": "structured", "weight": 0.6, "query": "4821"}, {"name": "temporal", "weight": 0.4, "query": "document 4821 latest changes"}], "reason": "identifier plus recency comparison", "confidence": 0.85}</output>
</example>
<example>
<query>what caused the incident linked to deploy-1832, and has that been fixed in a later release</query>
<output>{"strategies": [{"name": "structured", "weight": 0.4, "query": "deploy-1832"}, {"name": "graph", "weight": 0.35, "query": "deploy-1832 incident cause"}, {"name": "temporal", "weight": 0.25, "query": "deploy-1832 fix later release"}], "reason": "identifier, causal relationship, and recency all needed", "confidence": 0.75}</output>
</example>
</examples>

Use only strategy names listed in ALLOWED_STRATEGIES in the user message.
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
        f"HAS_DOCUMENT_FILTER: {filters.document_id is not None or filters.document_version_id is not None}\n"
        f"CURRENT_DATE: {datetime.now(UTC).date().isoformat()}"
    )


def _plan_from_result(
    result: RouterPlanSchema,
    *,
    top_k: int,
    allowed_strategies: frozenset[str],
) -> RetrievalPlan | None:
    selected = [
        StrategySelection(
            name=item.name,
            weight=item.weight,
            top_k=top_k,
            query=item.query,
            as_of=item.as_of,
        )
        for item in result.strategies
        if item.name in allowed_strategies
    ]
    selected = sorted(selected, key=lambda item: item.weight, reverse=True)
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
