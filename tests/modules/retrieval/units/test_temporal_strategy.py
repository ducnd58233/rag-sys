from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.plan import RetrievalPlan, RouterKind
from src.modules.retrieval.infra.elasticsearch.helper import build_filter_clauses
from src.modules.retrieval.infra.strategies.temporal import TemporalStrategy


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
                        "_id": "chunk-1",
                        "_score": 3.5,
                        "_source": {
                            "chunk_id": "chunk-1",
                            "document_id": "doc-1",
                            "content": "latest attention formula",
                            "metadata": {"filename": "paper.pdf"},
                        },
                    }
                ]
            }
        }


def test_build_filter_clauses_applies_current_validity() -> None:
    as_of = datetime(2026, 7, 20, tzinfo=timezone.utc)

    clauses = build_filter_clauses(RetrievalFilter(org_id=7, as_of=as_of))

    assert clauses == [
        {"term": {"org_id": "7"}},
        {"range": {"valid_from": {"lte": "2026-07-20T00:00:00+00:00"}}},
        {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "must_not": [
                                {"exists": {"field": "valid_to"}},
                            ],
                        }
                    },
                    {"range": {"valid_to": {"gt": "2026-07-20T00:00:00+00:00"}}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]


def test_build_filter_clauses_skips_validity_for_exact_version() -> None:
    clauses = build_filter_clauses(
        RetrievalFilter(
            org_id=7,
            document_id=11,
            document_version_id=13,
        )
    )

    assert clauses == [
        {"term": {"org_id": "7"}},
        {"term": {"document_id": "11"}},
        {"term": {"document_version_id": "13"}},
    ]


@pytest.mark.asyncio
async def test_temporal_strategy_applies_acl_as_of_and_recency_decay() -> None:
    es = FakeElasticsearch()
    strategy = TemporalStrategy(
        es,
        SimpleNamespace(index="rag-documents"),
        SimpleNamespace(temporal_recency_half_life_days=30.0),
    )
    as_of = datetime(2026, 7, 20, tzinfo=timezone.utc)

    hits = await strategy.retrieve(
        "latest attention",
        plan=RetrievalPlan.single(
            strategy="temporal",
            top_k=5,
            router_kind=RouterKind.RULE,
            reason="temporal",
        ),
        filters=RetrievalFilter(org_id=7, as_of=as_of),
    )

    request = es.client.search_request
    assert request is not None
    assert request["index"] == "rag-documents"
    assert request["size"] == 5
    query = request["query"]
    assert query["function_score"]["query"]["bool"]["filter"] == build_filter_clauses(
        RetrievalFilter(org_id=7, as_of=as_of),
    )
    assert query["function_score"]["functions"][0]["gauss"]["valid_from"] == {
        "origin": "2026-07-20T00:00:00+00:00",
        "scale": "30.0d",
        "decay": 0.5,
    }
    assert hits[0].chunk_id == "chunk-1"
