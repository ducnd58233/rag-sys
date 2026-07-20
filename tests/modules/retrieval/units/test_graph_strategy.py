from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.plan import RetrievalPlan, RouterKind, StrategySelection
from src.modules.retrieval.infra.strategies.graph import GraphStrategy


class FakeElasticsearch:
    def __init__(self) -> None:
        self.client = FakeElasticsearchClient()


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.search_request: dict[str, object] | None = None

    async def search(self, **kwargs: object) -> dict[str, object]:
        self.search_request = kwargs
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 2.0,
                        "_source": {
                            "chunk_id": "chunk-1",
                            "document_id": "101",
                            "content": "checkout depends on payment",
                            "metadata": {},
                        },
                    }
                ]
            }
        }


class FakeGraphSearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Sequence[str], int, int]] = []

    async def related_document_ids(
        self,
        *,
        org_id: int,
        entities: Sequence[str],
        max_hops: int,
        limit: int,
    ) -> Sequence[str]:
        self.calls.append((org_id, entities, max_hops, limit))
        return ("101", "102")


class FakeGraphQueryAnalyzer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def analyze(self, query: str) -> Sequence[str]:
        self.calls.append(query)
        return ("checkout",)


@pytest.mark.asyncio
async def test_graph_strategy_resolves_related_documents_through_elasticsearch() -> None:
    elasticsearch = FakeElasticsearch()
    graph_searcher = FakeGraphSearcher()
    query_analyzer = FakeGraphQueryAnalyzer()
    strategy = GraphStrategy(
        elasticsearch,
        SimpleNamespace(index="rag-documents"),
        SimpleNamespace(graph_max_hops=2),
        graph_searcher,
        query_analyzer,
    )

    hits = await strategy.retrieve(
        "what depends on checkout",
        plan=RetrievalPlan(
            strategies=(
                StrategySelection(
                    name="graph",
                    weight=1.0,
                    top_k=5,
                    query="checkout dependencies",
                ),
            ),
            router_kind=RouterKind.LLM,
            reason="relationship",
            confidence=0.8,
        ),
        filters=RetrievalFilter(org_id=7),
    )

    assert query_analyzer.calls == ["checkout dependencies"]
    assert graph_searcher.calls == [(7, ("checkout",), 2, 5)]
    query = elasticsearch.client.search_request["query"]
    assert {"terms": {"document_id": ["101", "102"]}} in query["bool"]["filter"]
    assert query["bool"]["must"][0]["match"]["content"]["query"] == (
        "checkout dependencies"
    )
    assert hits[0].chunk_id == "chunk-1"
