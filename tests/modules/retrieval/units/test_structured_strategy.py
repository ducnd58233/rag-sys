from collections.abc import Sequence

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter, RetrieveRequest
from src.modules.retrieval.app.router import CompositeQueryRouter, RuleRouter
from src.modules.retrieval.app.strategy_registry import RetrievalStrategyRegistry
from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan, RouterKind
from src.modules.retrieval.infra.fusion.reciprocal_rank import ReciprocalRankFusion
from src.modules.retrieval.infra.strategies import (
    LexicalStrategy,
    SemanticStrategy,
    StructuredStrategy,
)
from src.shared.configs.settings import (
    ElasticsearchSettings,
    RetrievalSettings,
    RoutingSettings,
)


class FakeElasticsearch:
    def __init__(self, hits: Sequence[dict[str, object]]) -> None:
        self.client = FakeElasticsearchClient(hits)


class FakeElasticsearchClient:
    def __init__(self, hits: Sequence[dict[str, object]]) -> None:
        self.searches: list[dict[str, object]] = []
        self._hits = tuple(hits)

    async def search(self, **kwargs: object) -> dict[str, object]:
        self.searches.append(kwargs)
        return {"hits": {"hits": list(self._hits)}}


class FakeEmbedder:
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3]]


class FakeLexicalSearcher:
    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        return (_hit("lexical", score=1.0),)


class FakeDenseSearcher:
    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        return (_hit("semantic", score=1.0),)


@pytest.mark.asyncio
async def test_structured_strategy_queries_explicit_filters_without_embedding() -> None:
    elasticsearch = FakeElasticsearch((_es_hit("deploy-1832"),))
    strategy = StructuredStrategy(
        elasticsearch,
        ElasticsearchSettings(index="test-index"),
        RoutingSettings(structured_fields=["chunk_id"]),
    )
    plan = RetrievalPlan.single(
        strategy="structured",
        top_k=3,
        router_kind=RouterKind.RULE,
        reason="explicit_filter",
    )

    result = await strategy.retrieve(
        "deploy-1832",
        plan=plan,
        filters=RetrievalFilter(org_id=1, document_id=42),
    )

    search = elasticsearch.client.searches[0]
    query = search["query"]
    assert search["index"] == "test-index"
    assert search["size"] == 3
    assert query["bool"]["must"] == [{"match_all": {}}]
    assert {"term": {"org_id": "1"}} in query["bool"]["filter"]
    assert {"term": {"document_id": "42"}} in query["bool"]["filter"]
    assert [hit.chunk_id for hit in result] == ["deploy-1832"]


@pytest.mark.asyncio
async def test_structured_strategy_queries_configured_fields_for_identifier() -> None:
    elasticsearch = FakeElasticsearch((_es_hit("deploy-1832"),))
    strategy = StructuredStrategy(
        elasticsearch,
        ElasticsearchSettings(index="test-index"),
        RoutingSettings(structured_fields=["chunk_id", "metadata.ticket"]),
    )
    plan = RetrievalPlan(
        strategies=(
            _selection(
                name="structured",
                top_k=5,
                query="deploy-1832",
            ),
        ),
        router_kind=RouterKind.RULE,
        reason="identifier",
        confidence=0.9,
    )

    await strategy.retrieve(
        "show deploy-1832 status",
        plan=plan,
        filters=RetrievalFilter(org_id=1),
    )

    query = elasticsearch.client.searches[0]["query"]
    should = query["bool"]["must"][0]["bool"]["should"]
    assert should == [
        {"term": {"chunk_id": "deploy-1832"}},
        {"term": {"metadata.ticket": "deploy-1832"}},
    ]


@pytest.mark.asyncio
async def test_identifier_retrieval_path_skips_embedding() -> None:
    retrieval_settings = RetrievalSettings(top_k=2)
    routing_settings = RoutingSettings(
        identifier_patterns=[r"deploy-\d+"],
        structured_fields=["chunk_id"],
    )
    elasticsearch = FakeElasticsearch((_es_hit("deploy-1832"),))
    embedder = FakeEmbedder()
    rank_fusion = ReciprocalRankFusion(rank_constant=retrieval_settings.rank_constant)
    structured = StructuredStrategy(
        elasticsearch,
        ElasticsearchSettings(index="test-index"),
        routing_settings,
    )
    lexical = LexicalStrategy(FakeLexicalSearcher(), retrieval_settings)
    semantic = SemanticStrategy(embedder, FakeDenseSearcher(), retrieval_settings)
    use_case = RetrieveUseCase(
        retrieval_settings,
        CompositeQueryRouter(routing_settings, RuleRouter(routing_settings)),
        RetrievalStrategyRegistry((structured, lexical, semantic)),
        rank_fusion,
    )

    result = await use_case.execute(
        RetrieveRequest(
            query="show deploy-1832 status",
            top_k=2,
            filters=RetrievalFilter(org_id=1),
        )
    )

    assert [strategy.name for strategy in result.plan.strategies] == [
        "structured",
        "lexical",
    ]
    assert embedder.calls == []
    assert len(elasticsearch.client.searches) == 1


def _selection(
    *,
    name: str,
    top_k: int,
    query: str,
):
    from src.modules.retrieval.domain.plan import StrategySelection

    return StrategySelection(name=name, weight=1.0, top_k=top_k, query=query)


def _es_hit(chunk_id: str) -> dict[str, object]:
    return {
        "_score": 2.0,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "content": "content",
            "metadata": {"ticket": chunk_id},
        },
    }


def _hit(chunk_id: str, *, score: float) -> HitChunk:
    return HitChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content {chunk_id}",
        score=score,
        metadata={},
    )
