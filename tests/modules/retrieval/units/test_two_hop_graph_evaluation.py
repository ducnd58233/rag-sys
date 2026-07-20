from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter, RetrieveRequest
from src.modules.retrieval.app.strategy_registry import RetrievalStrategyRegistry
from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.modules.retrieval.domain.models import GraphPathEvidence, HitChunk
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
from src.modules.retrieval.infra.strategies.graph import GraphStrategy
from src.shared.configs.settings import RetrievalSettings

# Scenario: "what does the checkout timeout depend on transitively" needs two
# graph hops (Checkout -> Payment -> DatabasePool) to reach the chunk that
# actually explains the root cause. Neither lexical nor semantic search share
# any vocabulary with that chunk, so hybrid alone returns nothing relevant.
# The graph strategy reaches it through traversal, and the evaluation asserts
# hybrid's blind spot and the graph strategy's coverage of it explicitly.


class FakeLexicalSearcher:
    def __init__(self, hits: Sequence[HitChunk]) -> None:
        self._hits = tuple(hits)

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        return self._hits


class FakeDenseSearcher:
    def __init__(self, hits: Sequence[HitChunk]) -> None:
        self._hits = tuple(hits)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        return self._hits


class FakeEmbedder:
    dimensions = 3

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeElasticsearch:
    def __init__(self, hits: Sequence[dict[str, object]]) -> None:
        self.client = FakeElasticsearchClient(hits)


class FakeElasticsearchClient:
    def __init__(self, hits: Sequence[dict[str, object]]) -> None:
        self._hits = hits

    async def search(self, **kwargs: object) -> dict[str, object]:
        return {"hits": {"hits": self._hits}}


class FakeGraphSearcher:
    def __init__(self, evidence: Sequence[GraphPathEvidence]) -> None:
        self._evidence = evidence

    async def related_evidence(
        self,
        *,
        org_id: int,
        entities: Sequence[str],
        max_hops: int,
        limit: int,
    ) -> Sequence[GraphPathEvidence]:
        return self._evidence


class FakeGraphQueryAnalyzer:
    async def analyze(self, query: str) -> Sequence[str]:
        return ("checkout",)


class FakeRouter:
    def __init__(self, plan: RetrievalPlan) -> None:
        self._plan = plan

    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan:
        return self._plan


def _root_cause_hit() -> dict[str, object]:
    return {
        "_score": 1.0,
        "_source": {
            "chunk_id": "chunk-db-pool",
            "document_id": "db-pool-doc",
            "content": (
                "The database connection pool has a hard cap of 20 connections, "
                "which starves requests under peak load."
            ),
            "metadata": {},
        },
    }


@pytest.mark.asyncio
async def test_hybrid_alone_misses_the_two_hop_root_cause_chunk() -> None:
    settings = RetrievalSettings(top_k=5)
    filters = RetrievalFilter(org_id=1)
    lexical = LexicalStrategy(FakeLexicalSearcher(()), settings)
    semantic = SemanticStrategy(FakeEmbedder(), FakeDenseSearcher(()), settings)
    rank_fusion = ReciprocalRankFusion(rank_constant=settings.rank_constant)
    hybrid = HybridStrategy(lexical, semantic, rank_fusion)
    plan = RetrievalPlan.hybrid(top_k=5, reason="test")

    hits = await hybrid.retrieve(
        "what does the checkout timeout depend on transitively",
        plan=plan,
        filters=filters,
    )

    assert hits == ()


@pytest.mark.asyncio
async def test_graph_strategy_reaches_the_root_cause_through_two_hops() -> None:
    settings = RetrievalSettings(top_k=5)
    filters = RetrievalFilter(org_id=1)
    lexical = LexicalStrategy(FakeLexicalSearcher(()), settings)
    semantic = SemanticStrategy(FakeEmbedder(), FakeDenseSearcher(()), settings)
    rank_fusion = ReciprocalRankFusion(rank_constant=settings.rank_constant)
    hybrid = HybridStrategy(lexical, semantic, rank_fusion)

    evidence = (
        GraphPathEvidence(
            document_id="db-pool-doc",
            chunk_ids=("chunk-db-pool",),
            anchor_entity="Checkout",
            related_entity="DatabasePool",
            relation_kinds=("depends_on", "depends_on"),
            hops=2,
        ),
    )
    graph = GraphStrategy(
        FakeElasticsearch(hits=[_root_cause_hit()]),
        SimpleNamespace(index="rag-documents"),
        SimpleNamespace(graph_max_hops=2),
        FakeGraphSearcher(evidence),
        FakeGraphQueryAnalyzer(),
    )

    registry = RetrievalStrategyRegistry((hybrid, graph))
    plan = RetrievalPlan(
        strategies=(
            StrategySelection(name="hybrid", weight=0.4, top_k=5),
            StrategySelection(
                name="graph",
                weight=0.6,
                top_k=5,
                query="checkout dependencies",
            ),
        ),
        router_kind=RouterKind.LLM,
        reason="relationship",
        confidence=0.85,
    )
    use_case = RetrieveUseCase(settings, FakeRouter(plan), registry, rank_fusion)

    result = await use_case.execute(
        RetrieveRequest(
            query="what does the checkout timeout depend on transitively",
            top_k=5,
            filters=filters,
        )
    )

    assert [item.chunk_id for item in result.items] == ["chunk-db-pool"]
    root_cause = result.items[0]
    assert root_cause.metadata["graph_evidence"] == "true"
    assert root_cause.metadata["graph_hops"] == "2"
    assert root_cause.metadata["graph_anchor_entity"] == "Checkout"
    assert root_cause.metadata["graph_related_entity"] == "DatabasePool"
