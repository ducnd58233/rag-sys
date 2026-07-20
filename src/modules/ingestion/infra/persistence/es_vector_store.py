import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from elasticsearch.helpers import async_bulk

from src.modules.ingestion.domain.errors import IngestionInternalError
from src.modules.ingestion.domain.models import Chunk, DocumentId
from src.shared.configs.settings import ElasticsearchSettings, EmbeddingSettings
from src.shared.infra.elasticsearch.client import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore:
    def __init__(
        self,
        elasticsearch: Elasticsearch,
        elasticsearch_settings: ElasticsearchSettings,
        embedding_settings: EmbeddingSettings,
    ) -> None:
        self._client = elasticsearch.client
        self._index = elasticsearch_settings.index
        self._shards = elasticsearch_settings.number_of_shards
        self._replicas = elasticsearch_settings.number_of_replicas
        self._dimensions = embedding_settings.dimensions

    async def create_index_if_not_exists(self) -> None:
        properties = {
            "org_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "document_version_id": {"type": "keyword"},
            "version_no": {"type": "integer"},
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "content": {"type": "text"},
            "metadata": {"type": "object", "enabled": True},
            "valid_from": {"type": "date"},
            "valid_to": {"type": "date"},
            "embedding": {
                "type": "dense_vector",
                "dims": self._dimensions,
                "index": True,
                "similarity": "cosine",
            },
            "indexed_at": {"type": "date"},
        }

        if await self._client.indices.exists(index=self._index):
            await self._client.indices.put_mapping(
                index=self._index,
                properties=properties,
            )
            await self._backfill_validity_fields()
            return

        await self._client.indices.create(
            index=self._index,
            settings={
                "number_of_shards": self._shards,
                "number_of_replicas": self._replicas,
            },
            mappings={
                "properties": properties,
            },
        )

    async def delete_by_document_version_id(
        self,
        *,
        org_id: int,
        document_version_id: int,
    ) -> None:
        await self._client.delete_by_query(
            index=self._index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"org_id": str(org_id)}},
                        {
                            "term": {
                                "document_version_id": str(document_version_id),
                            },
                        },
                    ],
                },
            },
            conflicts="proceed",
            refresh=True,
        )

    async def close_superseded_versions(
        self,
        *,
        org_id: int,
        document_id: DocumentId,
        active_document_version_id: int,
        valid_to: datetime,
    ) -> None:
        await self._client.update_by_query(
            index=self._index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"org_id": str(org_id)}},
                        {"term": {"document_id": str(document_id.value)}},
                        {"range": {"valid_from": {"lt": _datetime_value(valid_to)}}},
                    ],
                    "must_not": [
                        {
                            "term": {
                                "document_version_id": str(
                                    active_document_version_id,
                                ),
                            }
                        },
                        {"exists": {"field": "valid_to"}},
                    ],
                },
            },
            script={
                "source": "ctx._source.valid_to = params.valid_to",
                "lang": "painless",
                "params": {"valid_to": _datetime_value(valid_to)},
            },
            conflicts="proceed",
            refresh=True,
        )

    async def upsert(
        self,
        *,
        org_id: int,
        document_id: DocumentId,
        document_version_id: int,
        version_no: int,
        metadata: dict[str, str],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        valid_from: datetime,
        valid_to: datetime | None,
    ) -> None:
        indexed_at = datetime.now(timezone.utc)
        serialized_valid_from = _datetime_value(valid_from)
        serialized_valid_to = (
            _datetime_value(valid_to) if valid_to is not None else None
        )
        actions = [
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {
                    "org_id": str(org_id),
                    "document_id": str(document_id.value),
                    "document_version_id": str(document_version_id),
                    "version_no": version_no,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.index,
                    "content": chunk.content,
                    "metadata": {**metadata, **chunk.metadata},
                    "valid_from": serialized_valid_from,
                    "valid_to": serialized_valid_to,
                    "embedding": list(vector),
                    "indexed_at": indexed_at,
                },
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        try:
            success_count, errors = await async_bulk(
                self._client,
                actions,
                refresh=True,
            )
        except Exception as exc:
            raise IngestionInternalError(f"Bulk index failed: {exc}") from exc
        if errors:
            raise IngestionInternalError(f"Bulk index returned errors: {errors}")
        logger.info(
            "Indexed document_id=%s chunks=%d success=%d",
            document_id.value,
            len(chunks),
            success_count,
        )

    async def _backfill_validity_fields(self) -> None:
        await self._client.update_by_query(
            index=self._index,
            query={
                "bool": {
                    "filter": [{"exists": {"field": "indexed_at"}}],
                    "must_not": [{"exists": {"field": "valid_from"}}],
                }
            },
            script={
                "source": "ctx._source.valid_from = ctx._source.indexed_at",
                "lang": "painless",
            },
            conflicts="proceed",
            refresh=True,
        )


def _datetime_value(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
