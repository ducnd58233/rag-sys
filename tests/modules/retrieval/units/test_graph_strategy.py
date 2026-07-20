from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.models import GraphPathEvidence
from src.modules.retrieval.domain.plan import (
    RetrievalPlan,
    RouterKind,
    StrategySelection,
)
from src.modules.retrieval.infra.strategies.graph import GraphStrategy


class FakeElasticsearch:
    def __init__(self, hits: Sequence[dict[str, object]]) -> None:
        self.client = FakeElasticsearchClient(hits)


class FakeElasticsearchClient:
    def __init__(self, hits: Sequence[dict[str, object]]) -> None:
        self._hits = hits
        self.search_request: dict[str, object] | None = None

    async def search(self, **kwargs: object) -> dict[str, object]:
        self.search_request = kwargs
        return {"hits": {"hits": self._hits}}


class FakeGraphSearcher:
    def __init__(self, evidence: Sequence[GraphPathEvidence]) -> None:
        self.calls: list[tuple[int, Sequence[str], int, int]] = []
        self._evidence = evidence

    async def related_evidence(
        self,
        *,
        org_id: int,
        entities: Sequence[str],
        max_hops: int,
        limit: int,
    ) -> Sequence[GraphPathEvidence]:
        self.calls.append((org_id, entities, max_hops, limit))
        return self._evidence


class FakeGraphQueryAnalyzer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def analyze(self, query: str) -> Sequence[str]:
        self.calls.append(query)
        return ("checkout",)


def _hit(
    *, chunk_id: str, document_id: str, content: str, score: float = 2.0
) -> dict[str, object]:
    return {
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": content,
            "metadata": {},
        },
    }


def _plan(query: str = "checkout dependencies", top_k: int = 5) -> RetrievalPlan:
    return RetrievalPlan(
        strategies=(
            StrategySelection(name="graph", weight=1.0, top_k=top_k, query=query),
        ),
        router_kind=RouterKind.LLM,
        reason="relationship",
        confidence=0.8,
    )


def _strategy(elasticsearch: FakeElasticsearch, graph_searcher: FakeGraphSearcher):
    return GraphStrategy(
        elasticsearch,
        SimpleNamespace(index="rag-documents"),
        SimpleNamespace(graph_max_hops=2),
        graph_searcher,
        FakeGraphQueryAnalyzer(),
    )


@pytest.mark.asyncio
async def test_graph_strategy_returns_nothing_without_evidence() -> None:
    elasticsearch = FakeElasticsearch(hits=[])
    strategy = _strategy(elasticsearch, FakeGraphSearcher(evidence=()))

    hits = await strategy.retrieve(
        "what depends on checkout",
        plan=_plan(),
        filters=RetrievalFilter(org_id=7),
    )

    assert hits == ()
    assert elasticsearch.client.search_request is None


@pytest.mark.asyncio
async def test_graph_strategy_fetches_exact_chunks_with_hop_based_score() -> None:
    elasticsearch = FakeElasticsearch(
        hits=[
            _hit(chunk_id="chunk-1", document_id="101", content="checkout depends"),
        ]
    )
    evidence = (
        GraphPathEvidence(
            document_id="101",
            chunk_ids=("chunk-1",),
            anchor_entity="Checkout",
            related_entity="Payment",
            relation_kinds=("depends_on",),
            hops=1,
        ),
    )
    strategy = _strategy(elasticsearch, FakeGraphSearcher(evidence=evidence))

    hits = await strategy.retrieve(
        "what depends on checkout",
        plan=_plan(),
        filters=RetrievalFilter(org_id=7),
    )

    query = elasticsearch.client.search_request["query"]
    assert query["bool"]["should"] == [{"terms": {"chunk_id": ["chunk-1"]}}]

    assert len(hits) == 1
    hit = hits[0]
    assert hit.chunk_id == "chunk-1"
    assert hit.score == pytest.approx(0.5)
    assert hit.metadata["graph_evidence"] == "true"
    assert hit.metadata["graph_hops"] == "1"
    assert hit.metadata["graph_anchor_entity"] == "Checkout"
    assert hit.metadata["graph_related_entity"] == "Payment"
    assert hit.metadata["graph_relation_kinds"] == "depends_on"
    assert hit.metadata["graph_evidence_precision"] == "chunk"


@pytest.mark.asyncio
async def test_graph_strategy_falls_back_to_document_scoped_search_without_chunk_ids() -> (
    None
):
    elasticsearch = FakeElasticsearch(
        hits=[_hit(chunk_id="chunk-9", document_id="101", content="legacy graph data")]
    )
    evidence = (
        GraphPathEvidence(
            document_id="101",
            chunk_ids=(),
            anchor_entity="Checkout",
            related_entity="Payment",
            relation_kinds=("depends_on",),
            hops=1,
        ),
    )
    strategy = _strategy(elasticsearch, FakeGraphSearcher(evidence=evidence))

    hits = await strategy.retrieve(
        "what depends on checkout",
        plan=_plan(),
        filters=RetrievalFilter(org_id=7),
    )

    query = elasticsearch.client.search_request["query"]
    should = query["bool"]["should"]
    assert len(should) == 1
    assert should[0]["bool"]["filter"] == [{"terms": {"document_id": ["101"]}}]
    assert should[0]["bool"]["must"][0]["match"]["content"]["query"] == (
        "checkout dependencies"
    )

    hit = hits[0]
    assert hit.score == pytest.approx(2.0)
    assert hit.metadata["graph_evidence"] == "true"
    assert hit.metadata["graph_evidence_precision"] == "document"


@pytest.mark.asyncio
async def test_graph_strategy_combines_chunk_and_document_evidence_in_one_query() -> (
    None
):
    elasticsearch = FakeElasticsearch(hits=[])
    evidence = (
        GraphPathEvidence(
            document_id="101",
            chunk_ids=("chunk-1",),
            anchor_entity="Checkout",
            related_entity="Payment",
            relation_kinds=("depends_on",),
            hops=1,
        ),
        GraphPathEvidence(
            document_id="202",
            chunk_ids=(),
            anchor_entity="Checkout",
            related_entity="Refund",
            relation_kinds=("depends_on",),
            hops=2,
        ),
    )
    strategy = _strategy(elasticsearch, FakeGraphSearcher(evidence=evidence))

    await strategy.retrieve(
        "what depends on checkout",
        plan=_plan(),
        filters=RetrievalFilter(org_id=7),
    )

    should = elasticsearch.client.search_request["query"]["bool"]["should"]
    assert {"terms": {"chunk_id": ["chunk-1"]}} in should
    assert any(
        clause.get("bool", {}).get("filter") == [{"terms": {"document_id": ["202"]}}]
        for clause in should
    )


@pytest.mark.asyncio
async def test_graph_strategy_prefers_lowest_hops_evidence_for_a_chunk() -> None:
    elasticsearch = FakeElasticsearch(
        hits=[_hit(chunk_id="chunk-1", document_id="101", content="checkout")]
    )
    evidence = (
        GraphPathEvidence(
            document_id="101",
            chunk_ids=("chunk-1",),
            anchor_entity="Checkout",
            related_entity="Refund",
            relation_kinds=("depends_on", "depends_on"),
            hops=2,
        ),
        GraphPathEvidence(
            document_id="101",
            chunk_ids=("chunk-1",),
            anchor_entity="Checkout",
            related_entity="Payment",
            relation_kinds=("depends_on",),
            hops=1,
        ),
    )
    strategy = _strategy(elasticsearch, FakeGraphSearcher(evidence=evidence))

    hits = await strategy.retrieve(
        "what depends on checkout",
        plan=_plan(),
        filters=RetrievalFilter(org_id=7),
    )

    assert hits[0].metadata["graph_hops"] == "1"
    assert hits[0].metadata["graph_related_entity"] == "Payment"
    assert hits[0].score == pytest.approx(0.5)
