from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from src.modules.retrieval.infra.graphdb.neo4j_graph_searcher import Neo4jGraphSearcher


class FakeGraphDb:
    def __init__(self, rows: Sequence[Mapping[str, object]] = ()) -> None:
        self.statement: str | None = None
        self.parameters: Mapping[str, object] | None = None
        self._rows = tuple(rows)

    async def initialize(self) -> None:
        raise AssertionError("initialize should not be called")

    async def write(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        raise AssertionError("write should not be called")

    async def read(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> Sequence[Mapping[str, object]]:
        self.statement = statement
        self.parameters = parameters
        return self._rows

    async def close(self) -> None:
        raise AssertionError("close should not be called")


@pytest.mark.asyncio
async def test_graph_searcher_filters_org_and_caps_hops_in_cypher() -> None:
    graphdb = FakeGraphDb(
        rows=(
            {
                "document_id": 42,
                "anchor_entity": "Checkout",
                "related_entity": "Payment",
                "relation_kinds": ["depends_on"],
                "chunk_ids": ["42:1:0"],
                "hops": 1,
            },
        )
    )
    searcher = Neo4jGraphSearcher(graphdb)

    evidence = await searcher.related_evidence(
        org_id=7,
        entities=("Checkout",),
        max_hops=3,
        limit=5,
    )

    assert len(evidence) == 1
    item = evidence[0]
    assert item.document_id == "42"
    assert item.anchor_entity == "Checkout"
    assert item.related_entity == "Payment"
    assert item.relation_kinds == ("depends_on",)
    assert item.chunk_ids == ("42:1:0",)
    assert item.hops == 1
    assert graphdb.parameters == {
        "org_id": 7,
        "entities": ["checkout"],
        "limit": 5,
    }
    assert "anchor.org_id = $org_id" in graphdb.statement
    assert "version.org_id = $org_id" in graphdb.statement
    assert "RELATED*0..3" in graphdb.statement


@pytest.mark.asyncio
async def test_graph_searcher_dedupes_chunk_ids_from_mentions_and_relations() -> None:
    graphdb = FakeGraphDb(
        rows=(
            {
                "document_id": 42,
                "anchor_entity": "Checkout",
                "related_entity": "Checkout",
                "relation_kinds": [],
                "chunk_ids": ["42:1:0", "42:1:0", "42:1:1"],
                "hops": 0,
            },
        )
    )
    searcher = Neo4jGraphSearcher(graphdb)

    evidence = await searcher.related_evidence(
        org_id=7,
        entities=("Checkout",),
        max_hops=2,
        limit=5,
    )

    assert evidence[0].chunk_ids == ("42:1:0", "42:1:1")
    assert evidence[0].relation_kinds == ()


@pytest.mark.asyncio
async def test_graph_searcher_returns_empty_without_a_read_when_no_entities() -> None:
    graphdb = FakeGraphDb()
    searcher = Neo4jGraphSearcher(graphdb)

    evidence = await searcher.related_evidence(
        org_id=7,
        entities=(),
        max_hops=2,
        limit=5,
    )

    assert evidence == ()
    assert graphdb.statement is None
