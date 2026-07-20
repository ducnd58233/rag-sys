from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.modules.document.domain.models import DocumentProcessingStatus
from src.modules.ingestion.domain.graph import DocumentGraph, GraphEntity, GraphRelation
from src.modules.ingestion.domain.models import DocumentId, DocumentVersionSource
from src.modules.ingestion.infra.graphdb.neo4j_document_graph_store import (
    Neo4jDocumentGraphStore,
)
from src.modules.retrieval.infra.graphdb.neo4j_graph_searcher import Neo4jGraphSearcher
from src.shared.infra.graphdb import Neo4jGraphDb

pytestmark = pytest.mark.integration


def _source(
    *,
    org_id: int,
    document_id: int,
    document_version_id: int,
    version_no: int = 1,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> DocumentVersionSource:
    return DocumentVersionSource(
        org_id=org_id,
        document_id=DocumentId(document_id),
        document_version_id=document_version_id,
        storage_object_id=document_version_id + 1,
        version_no=version_no,
        bucket="documents",
        object_key="paper.pdf",
        filename="paper.pdf",
        mime_type="application/pdf",
        processing_status=DocumentProcessingStatus.PARSING,
        size_bytes=1000,
        checksum_sha256=None,
        valid_from=valid_from or datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=valid_to,
    )


@pytest.mark.asyncio
async def test_upsert_writes_entities_relations_and_chunk_ids_to_real_neo4j(
    graphdb: Neo4jGraphDb,
    unique_id: int,
) -> None:
    org_id = unique_id
    store = Neo4jDocumentGraphStore(graphdb)

    await store.upsert(
        source=_source(
            org_id=org_id, document_id=unique_id + 1, document_version_id=unique_id + 2
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(
                    name=f"Checkout-{unique_id}", kind="service", chunk_ids=("c1", "c2")
                ),
                GraphEntity(
                    name=f"Payment-{unique_id}", kind="service", chunk_ids=("c2",)
                ),
            ),
            relations=(
                GraphRelation(
                    source=f"Checkout-{unique_id}",
                    target=f"Payment-{unique_id}",
                    kind="depends_on",
                    chunk_ids=("c2",),
                ),
            ),
        ),
    )

    rows = await graphdb.read(
        """
        MATCH (a:Entity {org_id: $org_id, name: $checkout})-[edge:RELATED]->(b:Entity {org_id: $org_id, name: $payment})
        RETURN edge.kind AS kind, edge.chunk_ids AS chunk_ids
        """,
        {
            "org_id": org_id,
            "checkout": f"Checkout-{unique_id}",
            "payment": f"Payment-{unique_id}",
        },
    )

    assert len(rows) == 1
    assert rows[0]["kind"] == "depends_on"
    assert sorted(rows[0]["chunk_ids"]) == ["c2"]


@pytest.mark.asyncio
async def test_upsert_accumulates_chunk_ids_across_separate_ingestions(
    graphdb: Neo4jGraphDb,
    unique_id: int,
) -> None:
    """The RELATED edge is global per org, not per document version. Two
    documents that both mention the same relationship must union their
    chunk_ids on the same edge rather than the second overwriting the
    first. This is the exact reduce()-based Cypher that a FakeGraphDb
    unit test can only check the shape of, never actually execute."""
    org_id = unique_id
    store = Neo4jDocumentGraphStore(graphdb)
    checkout = f"Checkout-{unique_id}"
    payment = f"Payment-{unique_id}"

    await store.upsert(
        source=_source(
            org_id=org_id, document_id=unique_id + 1, document_version_id=unique_id + 2
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(name=checkout, kind="service", chunk_ids=("doc1:c1",)),
                GraphEntity(name=payment, kind="service", chunk_ids=("doc1:c1",)),
            ),
            relations=(
                GraphRelation(
                    source=checkout,
                    target=payment,
                    kind="depends_on",
                    chunk_ids=("doc1:c1",),
                ),
            ),
        ),
    )
    await store.upsert(
        source=_source(
            org_id=org_id, document_id=unique_id + 3, document_version_id=unique_id + 4
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(name=checkout, kind="service", chunk_ids=("doc2:c9",)),
                GraphEntity(name=payment, kind="service", chunk_ids=("doc2:c9",)),
            ),
            relations=(
                GraphRelation(
                    source=checkout,
                    target=payment,
                    kind="depends_on",
                    chunk_ids=("doc2:c9",),
                ),
            ),
        ),
    )

    rows = await graphdb.read(
        """
        MATCH (a:Entity {org_id: $org_id, name: $checkout})-[edge:RELATED]->(b:Entity {org_id: $org_id, name: $payment})
        RETURN edge.chunk_ids AS chunk_ids
        """,
        {"org_id": org_id, "checkout": checkout, "payment": payment},
    )

    assert len(rows) == 1
    assert sorted(rows[0]["chunk_ids"]) == ["doc1:c1", "doc2:c9"]


@pytest.mark.asyncio
async def test_upsert_creates_supersedes_edge_for_second_version(
    graphdb: Neo4jGraphDb,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1
    store = Neo4jDocumentGraphStore(graphdb)

    await store.upsert(
        source=_source(
            org_id=org_id,
            document_id=document_id,
            document_version_id=unique_id + 2,
            version_no=1,
        ),
        graph=DocumentGraph(entities=(), relations=()),
    )
    await store.upsert(
        source=_source(
            org_id=org_id,
            document_id=document_id,
            document_version_id=unique_id + 3,
            version_no=2,
        ),
        graph=DocumentGraph(entities=(), relations=()),
    )

    rows = await graphdb.read(
        """
        MATCH (current:Version {org_id: $org_id, document_id: $document_id, version_no: 2})
              -[:SUPERSEDES]->
              (previous:Version {org_id: $org_id, document_id: $document_id, version_no: 1})
        RETURN count(*) AS edge_count
        """,
        {"org_id": org_id, "document_id": document_id},
    )
    assert rows[0]["edge_count"] == 1


@pytest.mark.asyncio
async def test_related_evidence_traverses_two_hops_with_correct_chunk_ids(
    graphdb: Neo4jGraphDb,
    unique_id: int,
) -> None:
    org_id = unique_id
    store = Neo4jDocumentGraphStore(graphdb)
    searcher = Neo4jGraphSearcher(graphdb)
    checkout = f"Checkout-{unique_id}"
    payment = f"Payment-{unique_id}"
    db_pool = f"DatabasePool-{unique_id}"

    # Checkout -> Payment -> DatabasePool, a genuine 2-hop chain.
    await store.upsert(
        source=_source(
            org_id=org_id, document_id=unique_id + 1, document_version_id=unique_id + 2
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(name=checkout, kind="service", chunk_ids=("c-checkout",)),
                GraphEntity(name=payment, kind="service", chunk_ids=("c-payment",)),
            ),
            relations=(
                GraphRelation(
                    source=checkout,
                    target=payment,
                    kind="depends_on",
                    chunk_ids=("c-checkout-payment",),
                ),
            ),
        ),
    )
    await store.upsert(
        source=_source(
            org_id=org_id, document_id=unique_id + 3, document_version_id=unique_id + 4
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(name=payment, kind="service", chunk_ids=("c-payment-2",)),
                GraphEntity(name=db_pool, kind="component", chunk_ids=("c-dbpool",)),
            ),
            relations=(
                GraphRelation(
                    source=payment,
                    target=db_pool,
                    kind="depends_on",
                    chunk_ids=("c-payment-dbpool",),
                ),
            ),
        ),
    )

    evidence = await searcher.related_evidence(
        org_id=org_id,
        entities=(checkout,),
        max_hops=2,
        limit=10,
    )

    by_related = {item.related_entity: item for item in evidence}
    assert db_pool in by_related
    root_cause = by_related[db_pool]
    assert root_cause.hops == 2
    assert root_cause.anchor_entity == checkout
    assert "c-payment-dbpool" in root_cause.chunk_ids


@pytest.mark.asyncio
async def test_related_evidence_does_not_leak_across_organizations(
    graphdb: Neo4jGraphDb,
    unique_id: int,
) -> None:
    org_a = unique_id
    org_b = unique_id + 1
    shared_name = f"Checkout-{unique_id}"
    store = Neo4jDocumentGraphStore(graphdb)
    searcher = Neo4jGraphSearcher(graphdb)

    await store.upsert(
        source=_source(
            org_id=org_a, document_id=unique_id + 10, document_version_id=unique_id + 11
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(
                    name=shared_name, kind="service", chunk_ids=("org-a-chunk",)
                ),
            ),
            relations=(),
        ),
    )
    await store.upsert(
        source=_source(
            org_id=org_b, document_id=unique_id + 20, document_version_id=unique_id + 21
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(
                    name=shared_name, kind="service", chunk_ids=("org-b-chunk",)
                ),
            ),
            relations=(),
        ),
    )

    evidence = await searcher.related_evidence(
        org_id=org_a, entities=(shared_name,), max_hops=1, limit=10
    )

    for item in evidence:
        assert "org-b-chunk" not in item.chunk_ids


@pytest.mark.asyncio
async def test_related_evidence_excludes_versions_past_their_valid_to(
    graphdb: Neo4jGraphDb,
    unique_id: int,
) -> None:
    org_id = unique_id
    store = Neo4jDocumentGraphStore(graphdb)
    searcher = Neo4jGraphSearcher(graphdb)
    entity_name = f"Deprecated-{unique_id}"

    await store.upsert(
        source=_source(
            org_id=org_id,
            document_id=unique_id + 1,
            document_version_id=unique_id + 2,
            valid_from=datetime.now(timezone.utc) - timedelta(days=30),
            valid_to=datetime.now(timezone.utc) - timedelta(days=1),
        ),
        graph=DocumentGraph(
            entities=(
                GraphEntity(
                    name=entity_name, kind="service", chunk_ids=("expired-chunk",)
                ),
            ),
            relations=(),
        ),
    )

    evidence = await searcher.related_evidence(
        org_id=org_id, entities=(entity_name,), max_hops=1, limit=10
    )

    assert evidence == ()
