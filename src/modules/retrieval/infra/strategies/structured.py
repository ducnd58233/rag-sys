from __future__ import annotations

from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import IRetrievalStrategy
from src.modules.retrieval.domain.errors import RetrievalInternalError
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan
from src.modules.retrieval.infra.elasticsearch.helper import (
    _metadata_as_str_map,
    build_filter_clauses,
)
from src.shared.configs.settings import ElasticsearchSettings, RoutingSettings
from src.shared.infra.elasticsearch.client import Elasticsearch


class StructuredStrategy(IRetrievalStrategy):
    def __init__(
        self,
        elasticsearch: Elasticsearch,
        elasticsearch_settings: ElasticsearchSettings,
        routing_settings: RoutingSettings,
    ) -> None:
        self._client = elasticsearch.client
        self._index = elasticsearch_settings.index
        self._fields = tuple(routing_settings.structured_fields)

    @property
    def name(self) -> str:
        return "structured"

    def supports(self, plan: RetrievalPlan) -> bool:
        return plan.selection_for(self.name) is not None

    async def retrieve(
        self,
        query: str,
        *,
        plan: RetrievalPlan,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        selection = plan.selection_for(self.name)
        if selection is None:
            return ()
        try:
            response = await self._client.search(
                index=self._index,
                size=selection.top_k,
                query=_query(selection.query or query, filters, self._fields),
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
            raise RetrievalInternalError(f"structured search failed: {e}") from e
        return tuple(
            _hit_chunk(hit) for hit in response.get("hits", {}).get("hits", [])
        )


def _query(
    query: str,
    filters: RetrievalFilter,
    fields: Sequence[str],
) -> dict[str, object]:
    if filters.document_id is not None or filters.document_version_id is not None:
        must: list[dict[str, object]] = [{"match_all": {}}]
    else:
        must = [
            {
                "bool": {
                    "should": [{"term": {field: query}} for field in fields],
                    "minimum_should_match": 1,
                }
            }
        ]
    return {
        "bool": {
            "must": must,
            "filter": build_filter_clauses(filters),
        }
    }


def _hit_chunk(hit: dict[str, object]) -> HitChunk:
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        source = {}
    return HitChunk(
        chunk_id=str(source.get("chunk_id", hit.get("_id", ""))),
        document_id=str(source.get("document_id", "")),
        content=str(source.get("content", "")),
        score=float(hit.get("_score", 0.0)),
        metadata=_metadata_as_str_map(source.get("metadata", {})),
    )
