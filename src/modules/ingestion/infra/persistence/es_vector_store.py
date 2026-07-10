from collections.abc import Sequence
from datetime import datetime, timezone
import logging

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
        if await self._client.indices.exists(index=self._index):
            return
        
        await self._client.indices.create(
            index=self._index,
            settings={
                "number_of_shards": self._shards,
                "number_of_replicas": self._replicas,
            },
            mappings={
                "properties": {
                    "document_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self._dimensions,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "indexed_at": {"type": "date"},
                },
            },
        )

    async def delete_by_document_id(self, document_id: DocumentId) -> None:
        await self._client.delete_by_query(
            index=self._index,
            query={"term": {"document_id": document_id.value}},
            conflicts="proceed",
            refresh=True,
        )

    async def upsert(
        self,
        document_id: DocumentId,
        metadata: dict[str, str],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        indexed_at = datetime.now(timezone.utc)
        actions = [
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {
                    "document_id": document_id.value,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.index,
                    "content": chunk.content,
                    "metadata": {**metadata, **chunk.metadata},
                    "embedding": list(vector),
                    "indexed_at": indexed_at,
                },
            } for chunk, vector in zip(chunks, vectors)
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
