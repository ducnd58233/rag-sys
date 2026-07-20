from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import (
    IGraphQueryAnalyzer,
    IGraphSearcher,
    IRetrievalStrategy,
)
from src.modules.retrieval.domain.errors import RetrievalInternalError
from src.modules.retrieval.domain.models import GraphPathEvidence, HitChunk
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
        evidence = await self._graph_searcher.related_evidence(
            org_id=filters.org_id,
            entities=entities,
            max_hops=self._max_hops,
            limit=selection.top_k,
        )
        if not evidence:
            return ()

        chunk_evidence = _best_evidence_by_chunk_id(evidence)
        document_evidence = _best_evidence_by_document_id(evidence)
        try:
            response = await self._client.search(
                index=self._index,
                size=selection.top_k,
                query=_query(graph_query, filters, chunk_evidence, document_evidence),
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
            _hit_chunk(hit, chunk_evidence, document_evidence)
            for hit in response.get("hits", {}).get("hits", [])
        )


def _best_evidence_by_chunk_id(
    evidence: Sequence[GraphPathEvidence],
) -> dict[str, GraphPathEvidence]:
    best: dict[str, GraphPathEvidence] = {}
    for item in evidence:
        for chunk_id in item.chunk_ids:
            existing = best.get(chunk_id)
            if existing is None or item.hops < existing.hops:
                best[chunk_id] = item
    return best


def _best_evidence_by_document_id(
    evidence: Sequence[GraphPathEvidence],
) -> dict[str, GraphPathEvidence]:
    best: dict[str, GraphPathEvidence] = {}
    for item in evidence:
        if item.chunk_ids:
            continue
        existing = best.get(item.document_id)
        if existing is None or item.hops < existing.hops:
            best[item.document_id] = item
    return best


def _query(
    query: str,
    filters: RetrievalFilter,
    chunk_evidence: Mapping[str, GraphPathEvidence],
    document_evidence: Mapping[str, GraphPathEvidence],
) -> dict[str, object]:
    should: list[dict[str, object]] = []
    if chunk_evidence:
        should.append({"terms": {"chunk_id": list(chunk_evidence)}})
    if document_evidence:
        should.append(_document_scoped_clause(query, list(document_evidence)))
    return {
        "bool": {
            "should": should,
            "minimum_should_match": 1,
            "filter": build_filter_clauses(filters),
        }
    }


def _document_scoped_clause(
    query: str,
    document_ids: Sequence[str],
) -> dict[str, object]:
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
            "filter": [{"terms": {"document_id": list(document_ids)}}],
        }
    }


def _hop_score(hops: int) -> float:
    return 1.0 / (1.0 + hops)


def _hit_chunk(
    hit: dict[str, object],
    chunk_evidence: Mapping[str, GraphPathEvidence],
    document_evidence: Mapping[str, GraphPathEvidence],
) -> HitChunk:
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        source = {}
    chunk_id = str(source.get("chunk_id", hit.get("_id", "")))
    document_id = str(source.get("document_id", ""))
    metadata = _metadata_as_str_map(source.get("metadata", {}))
    score = float(hit.get("_score", 0.0))

    evidence = chunk_evidence.get(chunk_id)
    precision = "chunk"
    if evidence is None:
        evidence = document_evidence.get(document_id)
        precision = "document"
    if evidence is not None:
        if precision == "chunk":
            score = _hop_score(evidence.hops)
        metadata = {
            **metadata,
            "graph_evidence": "true",
            "graph_evidence_precision": precision,
            "graph_hops": str(evidence.hops),
            "graph_anchor_entity": evidence.anchor_entity,
            "graph_related_entity": evidence.related_entity,
            "graph_relation_kinds": "|".join(evidence.relation_kinds),
        }

    return HitChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=str(source.get("content", "")),
        score=score,
        metadata=metadata,
    )
