from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from src.modules.retrieval.infra.graphdb.neo4j_graph_searcher import Neo4jGraphSearcher


class FakeGraphDb:
    def __init__(self) -> None:
        self.statement: str | None = None
        self.parameters: Mapping[str, object] | None = None

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
        return ({"document_id": 42},)

    async def close(self) -> None:
        raise AssertionError("close should not be called")


@pytest.mark.asyncio
async def test_graph_searcher_filters_org_and_caps_hops_in_cypher() -> None:
    graphdb = FakeGraphDb()
    searcher = Neo4jGraphSearcher(graphdb)

    document_ids = await searcher.related_document_ids(
        org_id=7,
        entities=("Checkout",),
        max_hops=3,
        limit=5,
    )

    assert document_ids == ("42",)
    assert graphdb.parameters == {
        "org_id": 7,
        "entities": ["checkout"],
        "limit": 5,
    }
    assert "anchor.org_id = $org_id" in graphdb.statement
    assert "version.org_id = $org_id" in graphdb.statement
    assert "RELATED*0..3" in graphdb.statement
