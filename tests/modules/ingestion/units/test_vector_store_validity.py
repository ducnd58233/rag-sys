from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.modules.ingestion.domain.models import Chunk, DocumentId
from src.modules.ingestion.infra.persistence import es_vector_store
from src.modules.ingestion.infra.persistence.es_vector_store import (
    ElasticsearchVectorStore,
)


class FakeElasticsearch:
    def __init__(self) -> None:
        self.client = FakeElasticsearchClient()


class FakeIndices:
    def __init__(self) -> None:
        self.properties: dict[str, object] | None = None

    async def exists(self, *, index: str) -> bool:
        return True

    async def put_mapping(
        self,
        *,
        index: str,
        properties: dict[str, object],
    ) -> None:
        self.properties = properties


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.deleted_queries: list[dict[str, object]] = []
        self.updated_queries: list[dict[str, object]] = []

    async def delete_by_query(self, **kwargs: object) -> None:
        self.deleted_queries.append(kwargs)

    async def update_by_query(self, **kwargs: object) -> None:
        self.updated_queries.append(kwargs)


@pytest.mark.asyncio
async def test_index_mapping_and_backfill_include_validity_fields() -> None:
    es = FakeElasticsearch()
    store = _store(es)

    await store.create_index_if_not_exists()

    assert es.client.indices.properties["valid_from"] == {"type": "date"}
    assert es.client.indices.properties["valid_to"] == {"type": "date"}
    assert es.client.updated_queries == [
        {
            "index": "rag-documents",
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": "indexed_at"}}],
                    "must_not": [{"exists": {"field": "valid_from"}}],
                }
            },
            "script": {
                "source": "ctx._source.valid_from = ctx._source.indexed_at",
                "lang": "painless",
            },
            "conflicts": "proceed",
            "refresh": True,
        }
    ]


@pytest.mark.asyncio
async def test_delete_targets_document_version() -> None:
    es = FakeElasticsearch()
    store = _store(es)

    await store.delete_by_document_version_id(org_id=7, document_version_id=13)

    assert es.client.deleted_queries == [
        {
            "index": "rag-documents",
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"org_id": "7"}},
                        {"term": {"document_version_id": "13"}},
                    ],
                },
            },
            "conflicts": "proceed",
            "refresh": True,
        }
    ]


@pytest.mark.asyncio
async def test_close_superseded_versions_sets_valid_to() -> None:
    es = FakeElasticsearch()
    store = _store(es)
    valid_to = datetime(2026, 7, 10, tzinfo=timezone.utc)

    await store.close_superseded_versions(
        org_id=7,
        document_id=DocumentId(11),
        active_document_version_id=102,
        valid_to=valid_to,
    )

    assert es.client.updated_queries == [
        {
            "index": "rag-documents",
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"org_id": "7"}},
                        {"term": {"document_id": "11"}},
                        {
                            "range": {
                                "valid_from": {"lt": "2026-07-10T00:00:00+00:00"}
                            }
                        },
                    ],
                    "must_not": [
                        {"term": {"document_version_id": "102"}},
                        {"exists": {"field": "valid_to"}},
                    ],
                },
            },
            "script": {
                "source": "ctx._source.valid_to = params.valid_to",
                "lang": "painless",
                "params": {"valid_to": "2026-07-10T00:00:00+00:00"},
            },
            "conflicts": "proceed",
            "refresh": True,
        }
    ]


@pytest.mark.asyncio
async def test_upsert_writes_non_overlapping_version_validity(monkeypatch) -> None:
    captured_actions: list[dict[str, object]] = []

    async def fake_bulk(
        client: object,
        actions: list[dict[str, object]],
        *,
        refresh: bool,
    ) -> tuple[int, list[object]]:
        captured_actions.extend(actions)
        return len(actions), []

    monkeypatch.setattr(es_vector_store, "async_bulk", fake_bulk)
    store = _store(FakeElasticsearch())
    first_from = datetime(2026, 7, 1, tzinfo=timezone.utc)
    second_from = datetime(2026, 7, 10, tzinfo=timezone.utc)

    await store.upsert(
        org_id=7,
        document_id=DocumentId(11),
        document_version_id=101,
        version_no=1,
        metadata={},
        chunks=(_chunk(11, 1),),
        vectors=([0.1, 0.2],),
        valid_from=first_from,
        valid_to=second_from,
    )
    await store.upsert(
        org_id=7,
        document_id=DocumentId(11),
        document_version_id=102,
        version_no=2,
        metadata={},
        chunks=(_chunk(11, 2),),
        vectors=([0.2, 0.3],),
        valid_from=second_from,
        valid_to=None,
    )

    first_source = captured_actions[0]["_source"]
    second_source = captured_actions[1]["_source"]
    assert first_source["valid_from"] == "2026-07-01T00:00:00+00:00"
    assert first_source["valid_to"] == "2026-07-10T00:00:00+00:00"
    assert second_source["valid_from"] == "2026-07-10T00:00:00+00:00"
    assert second_source["valid_to"] is None


def _store(es: FakeElasticsearch) -> ElasticsearchVectorStore:
    return ElasticsearchVectorStore(
        es,
        SimpleNamespace(
            index="rag-documents",
            number_of_shards=1,
            number_of_replicas=0,
        ),
        SimpleNamespace(dimensions=2),
    )


def _chunk(document_id: int, version_no: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:{version_no}:0",
        document_id=DocumentId(document_id),
        index=0,
        content="content",
        metadata={},
    )
