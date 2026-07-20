from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from src.modules.document.domain.models import DocumentProcessingStatus
from src.modules.ingestion.domain.graph import DocumentGraph
from src.modules.ingestion.domain.models import DocumentId, DocumentVersionSource
from src.modules.ingestion.infra.graphdb.neo4j_document_graph_store import (
    Neo4jDocumentGraphStore,
)


class FakeGraphDb:
    def __init__(self) -> None:
        self.writes: list[tuple[str, Mapping[str, object] | None]] = []

    async def initialize(self) -> None:
        raise AssertionError("initialize should not be called")

    async def write(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        self.writes.append((statement, parameters))

    async def read(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> Sequence[Mapping[str, object]]:
        raise AssertionError("read should not be called")

    async def close(self) -> None:
        raise AssertionError("close should not be called")


@pytest.mark.asyncio
async def test_graph_store_writes_version_without_extracted_entities() -> None:
    graphdb = FakeGraphDb()
    store = Neo4jDocumentGraphStore(graphdb)

    await store.upsert(
        source=_source(version_no=2),
        graph=DocumentGraph(entities=(), relations=()),
    )

    assert len(graphdb.writes) == 3
    first_statement, first_parameters = graphdb.writes[0]
    supersedes_statement, supersedes_parameters = graphdb.writes[2]
    assert "MERGE (d:Document" in first_statement
    assert "MERGE (v:Version" in first_statement
    assert first_parameters["entities"] == []
    assert "MERGE (current)-[:SUPERSEDES]->(previous)" in supersedes_statement
    assert supersedes_parameters["previous_version_no"] == 1


def _source(*, version_no: int) -> DocumentVersionSource:
    return DocumentVersionSource(
        org_id=7,
        document_id=DocumentId(42),
        document_version_id=100 + version_no,
        storage_object_id=200 + version_no,
        version_no=version_no,
        bucket="documents",
        object_key="paper.pdf",
        filename="paper.pdf",
        mime_type="application/pdf",
        processing_status=DocumentProcessingStatus.PARSING,
        size_bytes=1000,
        checksum_sha256=None,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_to=None,
    )
