from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter, RetrieveRequest
from src.modules.retrieval.app.strategy_registry import RetrievalStrategyRegistry
from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)
from src.modules.retrieval.infra.fusion.reciprocal_rank import ReciprocalRankFusion
from src.modules.retrieval.infra.strategies import (
    HybridStrategy,
    LexicalStrategy,
    SemanticStrategy,
)
from src.shared.configs.settings import RetrievalSettings


class FakeRouter:
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


class FakeEmbedder:
    dimensions = 3

    def __init__(self, vector: list[float]) -> None:
        self.calls: list[Sequence[str]] = []
        self._vector = vector

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector]


class FakeLexicalSearcher:
    def __init__(self, hits: Sequence[HitChunk]) -> None:
        self.calls: list[tuple[str, int, RetrievalFilter]] = []
        self._hits = tuple(hits)

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        self.calls.append((query, limit, filters))
        return self._hits


class FakeDenseSearcher:
    def __init__(self, hits: Sequence[HitChunk]) -> None:
        self.calls: list[tuple[Sequence[float], int, int, RetrievalFilter]] = []
        self._hits = tuple(hits)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        self.calls.append((query_vector, limit, num_candidates, filters))
        return self._hits


@pytest.mark.asyncio
async def test_hybrid_strategy_matches_current_rrf_path() -> None:
    settings = RetrievalSettings(top_k=2, candidate_k=8, num_candidates=10)
    filters = RetrievalFilter(org_id=1)
    lexical_hits = (
        _hit("a", score=2.0),
        _hit("b", score=1.0),
    )
    semantic_hits = (
        _hit("b", score=3.0),
        _hit("c", score=2.0),
    )
    rank_fusion = ReciprocalRankFusion(rank_constant=settings.rank_constant)
    lexical = LexicalStrategy(FakeLexicalSearcher(lexical_hits), settings)
    semantic = SemanticStrategy(
        FakeEmbedder([0.1, 0.2, 0.3]),
        FakeDenseSearcher(semantic_hits),
        settings,
    )
    strategy = HybridStrategy(lexical, semantic, rank_fusion)
    plan = RetrievalPlan.hybrid(top_k=2, reason="test")

    result = await strategy.retrieve("query", plan=plan, filters=filters)

    assert result == rank_fusion.fuse([lexical_hits, semantic_hits], top_k=2)


@pytest.mark.asyncio
async def test_retrieve_use_case_dispatches_through_hybrid_strategy() -> None:
    settings = RetrievalSettings(top_k=2, candidate_k=8, num_candidates=10)
    filters = RetrievalFilter(org_id=1)
    lexical_hits = (_hit("a", score=2.0),)
    semantic_hits = (_hit("b", score=3.0),)
    embedder = FakeEmbedder([0.1, 0.2, 0.3])
    rank_fusion = ReciprocalRankFusion(rank_constant=settings.rank_constant)
    lexical = LexicalStrategy(FakeLexicalSearcher(lexical_hits), settings)
    semantic = SemanticStrategy(embedder, FakeDenseSearcher(semantic_hits), settings)
    registry = RetrievalStrategyRegistry(
        (HybridStrategy(lexical, semantic, rank_fusion), lexical, semantic)
    )
    plan = RetrievalPlan.hybrid(top_k=2, reason="test")
    router = FakeRouter(plan)
    use_case = RetrieveUseCase(settings, router, registry, rank_fusion)

    result = await use_case.execute(
        RetrieveRequest(query=" query ", top_k=2, filters=filters)
    )

    assert result.query == "query"
    assert result.plan.to_dict() == plan.to_dict()
    assert [item.chunk_id for item in result.items] == ["a", "b"]
    assert embedder.calls == [["query"]]
    assert router.calls == [("query", filters, 2)]


@pytest.mark.asyncio
async def test_retrieve_use_case_limits_single_strategy_results_to_top_k() -> None:
    settings = RetrievalSettings(top_k=2, num_candidates=10)
    filters = RetrievalFilter(org_id=1)
    lexical_searcher = FakeLexicalSearcher(
        (
            _hit("a", score=3.0),
            _hit("b", score=2.0),
            _hit("c", score=1.0),
        )
    )
    lexical = LexicalStrategy(lexical_searcher, settings)
    rank_fusion = ReciprocalRankFusion(rank_constant=settings.rank_constant)
    plan = RetrievalPlan.single(
        strategy="lexical",
        top_k=2,
        router_kind=RouterKind.RULE,
        reason="test",
    )
    use_case = RetrieveUseCase(
        settings,
        FakeRouter(plan),
        RetrievalStrategyRegistry((lexical,)),
        rank_fusion,
    )

    result = await use_case.execute(
        RetrieveRequest(query="query", top_k=2, filters=filters)
    )

    assert lexical_searcher.calls == [("query", 10, filters)]
    assert [item.chunk_id for item in result.items] == ["a", "b"]


def test_strategy_registry_selects_supported_strategies() -> None:
    settings = RetrievalSettings()
    lexical = LexicalStrategy(FakeLexicalSearcher(()), settings)
    semantic = SemanticStrategy(FakeEmbedder([0.1]), FakeDenseSearcher(()), settings)
    registry = RetrievalStrategyRegistry((lexical, semantic))
    plan = RetrievalPlan(
        strategies=(StrategySelection(name="semantic", weight=1.0, top_k=5),),
        router_kind=RouterKind.RULE,
        reason="semantic",
        confidence=1.0,
    )

    assert registry.select(plan) == (semantic,)


def _hit(chunk_id: str, *, score: float) -> HitChunk:
    return HitChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content {chunk_id}",
        score=score,
        metadata={},
    )
