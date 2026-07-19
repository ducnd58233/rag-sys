from __future__ import annotations

import logging
from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.errors import RetrievalInternalError
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.infra.elasticsearch.helper import (
    _metadata_as_str_map,
    build_filter_clauses,
)
from src.shared.configs.settings import ElasticsearchSettings
from src.shared.infra.elasticsearch.client import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchDenseSearcher:
    def __init__(self, es: Elasticsearch, settings: ElasticsearchSettings) -> None:
        self._client = es.client
        self._index = settings.index

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        knn: dict[str, object] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": limit,
            "num_candidates": max(num_candidates, limit),
        }

        knn["filter"] = build_filter_clauses(filters)

        try:
            response = await self._client.search(
                index=self._index,
                knn=knn,
                size=limit,
                source={
                    "includes": [
                        "chunk_id",
                        "document_id",
                        "content",
                        "metadata",
                    ],
                },
            )
        except Exception as e:
            raise RetrievalInternalError(f"dense search failed: {e}") from e

        hits = response.get("hits", {}).get("hits", [])
        results: list[HitChunk] = []

        for hit in hits:
            src = hit.get("_source", {})
            results.append(
                HitChunk(
                    chunk_id=src.get("chunk_id", hit.get("_id", "")),
                    document_id=src.get("document_id", ""),
                    content=src.get("content", ""),
                    score=float(hit.get("_score", 0.0)),
                    metadata=_metadata_as_str_map(src.get("metadata", {})),
                )
            )

        logger.info("dense search results: %d hits", len(results))
        return tuple(results)
