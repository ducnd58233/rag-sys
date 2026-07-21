from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

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


class TemporalStrategy(IRetrievalStrategy):
    def __init__(
        self,
        elasticsearch: Elasticsearch,
        elasticsearch_settings: ElasticsearchSettings,
        routing_settings: RoutingSettings,
    ) -> None:
        self._client = elasticsearch.client
        self._index = elasticsearch_settings.index
        self._half_life_days = routing_settings.temporal_recency_half_life_days

    @property
    def name(self) -> str:
        return "temporal"

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
                query=self._query(
                    selection.query or query,
                    filters,
                    as_of=_parse_as_of(selection.as_of),
                ),
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
            raise RetrievalInternalError(f"temporal search failed: {e}") from e
        return tuple(
            _hit_chunk(hit) for hit in response.get("hits", {}).get("hits", [])
        )

    def _query(
        self,
        query: str,
        filters: RetrievalFilter,
        as_of: datetime | None,
    ) -> dict[str, object]:
        origin = as_of or filters.as_of or datetime.now(timezone.utc)
        return {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "minimum_should_match": "70%",
                                    }
                                }
                            }
                        ],
                        "filter": build_filter_clauses(filters),
                    }
                },
                "functions": [
                    {
                        "gauss": {
                            "valid_from": {
                                "origin": origin.astimezone(
                                    timezone.utc,
                                ).isoformat(),
                                "scale": _decay_scale(self._half_life_days),
                                "decay": 0.5,
                            }
                        }
                    }
                ],
                "score_mode": "multiply",
                "boost_mode": "multiply",
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


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _decay_scale(half_life_days: float) -> str:
    hours = max(round(half_life_days * 24), 1)
    return f"{hours}h"
