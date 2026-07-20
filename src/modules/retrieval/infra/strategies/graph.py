from __future__ import annotations

from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import (
    IGraphQueryAnalyzer,
    IGraphSearcher,
    IRetrievalStrategy,
)
from src.modules.retrieval.domain.errors import RetrievalInternalError
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan
from src.modules.retrieval.infra.elasticsearch.helper import (
    _metadata_as_str_map,
    build_filter_clauses,
)
from src.shared.configs.settings import ElasticsearchSettings, RoutingSettings
from src.shared.infra.elasticsearch.client import Elasticsearch


class GraphStrategy(IRetrievalStrategy):
    def __init__(
        self,
        elasticsearch: Elasticsearch,
        elasticsearch_settings: ElasticsearchSettings,
        routing_settings: RoutingSettings,
        graph_searcher: IGraphSearcher,
        query_analyzer: IGraphQueryAnalyzer,
    ) -> None:
        self._client = elasticsearch.client
        self._index = elasticsearch_settings.index
        self._max_hops = routing_settings.graph_max_hops
        self._graph_searcher = graph_searcher
        self._query_analyzer = query_analyzer

    @property
    def name(self) -> str:
        return "graph"

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
        graph_query = selection.query or query
        entities = await self._query_analyzer.analyze(graph_query)
        document_ids = await self._graph_searcher.related_document_ids(
            org_id=filters.org_id,
            entities=entities,
            max_hops=self._max_hops,
            limit=selection.top_k,
        )
        if not document_ids:
            return ()
        try:
            response = await self._client.search(
                index=self._index,
                size=selection.top_k,
                query=_query(graph_query, filters, document_ids),
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
            raise RetrievalInternalError(f"graph search failed: {e}") from e
        return tuple(
            _hit_chunk(hit) for hit in response.get("hits", {}).get("hits", [])
        )


def _query(
    query: str,
    filters: RetrievalFilter,
    document_ids: Sequence[str],
) -> dict[str, object]:
    clauses = build_filter_clauses(filters)
    clauses.append({"terms": {"document_id": list(document_ids)}})
    return {
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
            "filter": clauses,
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
