from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from src.modules.ingestion.domain.models import Chunk, DocumentId
from src.modules.ingestion.infra.persistence.es_vector_store import (
    ElasticsearchVectorStore,
)
from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.infra.elasticsearch.dense_searcher import (
    ElasticsearchDenseSearcher,
)
from src.modules.retrieval.infra.elasticsearch.lexical_searcher import (
    ElasticsearchLexicalSearcher,
)
from src.shared.configs.settings import ElasticsearchSettings, EmbeddingSettings
from src.shared.infra.elasticsearch.client import Elasticsearch

pytestmark = pytest.mark.integration

_DIMENSIONS = 32


@pytest.fixture
def test_index_name(unique_id: int) -> str:
    # Every test gets its own index so tests never see each other's data,
    # even though the underlying container is shared for the whole session.
    return f"rag-test-{unique_id}"


@pytest.fixture
def test_index_settings(
    elasticsearch_settings: ElasticsearchSettings,
    test_index_name: str,
) -> ElasticsearchSettings:
    return elasticsearch_settings.model_copy(update={"index": test_index_name})


@pytest_asyncio.fixture
async def vector_store(
    elasticsearch: Elasticsearch,
    test_index_settings: ElasticsearchSettings,
    test_index_name: str,
) -> AsyncIterator[ElasticsearchVectorStore]:
    store = ElasticsearchVectorStore(
        elasticsearch,
        test_index_settings,
        EmbeddingSettings(dimensions=_DIMENSIONS),
    )
    await store.create_index_if_not_exists()
    try:
        yield store
    finally:
        await elasticsearch.client.indices.delete(
            index=test_index_name,
            ignore_unavailable=True,
        )


@pytest.fixture
def lexical_searcher(
    elasticsearch: Elasticsearch,
    test_index_settings: ElasticsearchSettings,
) -> ElasticsearchLexicalSearcher:
    return ElasticsearchLexicalSearcher(elasticsearch, test_index_settings)


@pytest.fixture
def dense_searcher(
    elasticsearch: Elasticsearch,
    test_index_settings: ElasticsearchSettings,
) -> ElasticsearchDenseSearcher:
    return ElasticsearchDenseSearcher(elasticsearch, test_index_settings)


def _vector(*head: float) -> list[float]:
    return [*head, *([0.0] * (_DIMENSIONS - len(head)))]


def _chunk(document_id: int, index: int, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:1:{index}",
        document_id=DocumentId(document_id),
        index=index,
        content=content,
        metadata={},
    )


@pytest.mark.asyncio
async def test_upsert_then_lexical_search_finds_indexed_content(
    vector_store: ElasticsearchVectorStore,
    lexical_searcher: ElasticsearchLexicalSearcher,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1

    await vector_store.upsert(
        org_id=org_id,
        document_id=DocumentId(document_id),
        document_version_id=unique_id + 2,
        version_no=1,
        metadata={},
        chunks=[_chunk(document_id, 0, "checkout depends on the payment service")],
        vectors=[_vector(0.1, 0.2, 0.3, 0.4)],
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
    )

    hits = await lexical_searcher.search(
        "payment service",
        limit=10,
        filters=RetrievalFilter(org_id=org_id),
    )

    assert len(hits) == 1
    assert hits[0].content == "checkout depends on the payment service"


@pytest.mark.asyncio
async def test_lexical_search_respects_org_id_filter(
    vector_store: ElasticsearchVectorStore,
    lexical_searcher: ElasticsearchLexicalSearcher,
    unique_id: int,
) -> None:
    org_id = unique_id
    other_org_id = unique_id + 100
    document_id = unique_id + 1

    await vector_store.upsert(
        org_id=org_id,
        document_id=DocumentId(document_id),
        document_version_id=unique_id + 2,
        version_no=1,
        metadata={},
        chunks=[_chunk(document_id, 0, "checkout dependency notes")],
        vectors=[_vector(0.1, 0.2, 0.3, 0.4)],
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
    )

    hits = await lexical_searcher.search(
        "checkout dependency",
        limit=10,
        filters=RetrievalFilter(org_id=other_org_id),
    )

    assert hits == ()


@pytest.mark.asyncio
async def test_dense_knn_search_returns_the_closest_vector(
    vector_store: ElasticsearchVectorStore,
    dense_searcher: ElasticsearchDenseSearcher,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1

    await vector_store.upsert(
        org_id=org_id,
        document_id=DocumentId(document_id),
        document_version_id=unique_id + 2,
        version_no=1,
        metadata={},
        chunks=[
            _chunk(document_id, 0, "near vector"),
            _chunk(document_id, 1, "far vector"),
        ],
        vectors=[_vector(1.0, 0.0), _vector(0.0, 0.0, 0.0, 1.0)],
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
    )

    hits = await dense_searcher.search(
        _vector(0.9, 0.1),
        limit=1,
        num_candidates=10,
        filters=RetrievalFilter(org_id=org_id),
    )

    assert len(hits) == 1
    assert hits[0].content == "near vector"


@pytest.mark.asyncio
async def test_delete_by_document_version_id_removes_chunks(
    vector_store: ElasticsearchVectorStore,
    lexical_searcher: ElasticsearchLexicalSearcher,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1
    document_version_id = unique_id + 2

    await vector_store.upsert(
        org_id=org_id,
        document_id=DocumentId(document_id),
        document_version_id=document_version_id,
        version_no=1,
        metadata={},
        chunks=[_chunk(document_id, 0, "to be deleted")],
        vectors=[_vector(0.1, 0.2, 0.3, 0.4)],
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
    )

    await vector_store.delete_by_document_version_id(
        org_id=org_id,
        document_version_id=document_version_id,
    )

    hits = await lexical_searcher.search(
        "deleted",
        limit=10,
        filters=RetrievalFilter(org_id=org_id),
    )
    assert hits == ()


@pytest.mark.asyncio
async def test_close_superseded_versions_sets_valid_to_on_real_data(
    elasticsearch: Elasticsearch,
    vector_store: ElasticsearchVectorStore,
    test_index_name: str,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1
    old_version_id = unique_id + 2
    new_version_id = unique_id + 3
    old_valid_from = datetime.now(timezone.utc) - timedelta(days=10)
    supersede_at = datetime.now(timezone.utc)

    await vector_store.upsert(
        org_id=org_id,
        document_id=DocumentId(document_id),
        document_version_id=old_version_id,
        version_no=1,
        metadata={},
        chunks=[_chunk(document_id, 0, "old content")],
        vectors=[_vector(0.1, 0.2, 0.3, 0.4)],
        valid_from=old_valid_from,
        valid_to=None,
    )

    await vector_store.close_superseded_versions(
        org_id=org_id,
        document_id=DocumentId(document_id),
        active_document_version_id=new_version_id,
        valid_to=supersede_at,
    )

    document = await elasticsearch.client.get(
        index=test_index_name,
        id=f"{document_id}:1:0",
    )
    assert document["_source"]["valid_to"] is not None


@pytest.mark.asyncio
async def test_backfill_populates_valid_from_from_indexed_at_on_legacy_data(
    elasticsearch: Elasticsearch,
    test_index_settings: ElasticsearchSettings,
    test_index_name: str,
    unique_id: int,
) -> None:
    """Simulates data indexed before valid_from existed: indexed_at is set,
    valid_from is absent. create_index_if_not_exists must backfill it when
    the index already exists, which is exactly the migration path this
    method exists for."""
    store = ElasticsearchVectorStore(
        elasticsearch,
        test_index_settings,
        EmbeddingSettings(dimensions=_DIMENSIONS),
    )
    await store.create_index_if_not_exists()
    try:
        await elasticsearch.client.index(
            index=test_index_name,
            id="legacy-chunk",
            document={
                "org_id": str(unique_id),
                "document_id": str(unique_id + 1),
                "document_version_id": str(unique_id + 2),
                "version_no": 1,
                "chunk_id": "legacy-chunk",
                "chunk_index": 0,
                "content": "legacy content with no valid_from",
                "metadata": {},
                "embedding": _vector(0.1, 0.2, 0.3, 0.4),
                "indexed_at": "2026-01-01T00:00:00+00:00",
            },
            refresh=True,
        )

        # Re-running against an existing index triggers the backfill path.
        await store.create_index_if_not_exists()

        document = await elasticsearch.client.get(
            index=test_index_name,
            id="legacy-chunk",
        )
        assert document["_source"]["valid_from"] == "2026-01-01T00:00:00+00:00"
    finally:
        await elasticsearch.client.indices.delete(
            index=test_index_name,
            ignore_unavailable=True,
        )
