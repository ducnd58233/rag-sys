from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from src.modules.retrieval.domain.models import GraphPathEvidence
from src.shared.app.ports import IGraphDb


class Neo4jGraphSearcher:
    def __init__(self, graphdb: IGraphDb) -> None:
        self._graphdb = graphdb

    async def related_evidence(
        self,
        *,
        org_id: int,
        entities: Sequence[str],
        max_hops: int,
        limit: int,
    ) -> Sequence[GraphPathEvidence]:
        if not entities:
            return ()
        rows = await self._graphdb.read(
            _query(max_hops),
            {
                "org_id": org_id,
                "entities": [entity.casefold() for entity in entities],
                "limit": limit,
            },
        )
        return tuple(_evidence_from_row(row) for row in rows)


def _query(max_hops: int) -> str:
    # RELATED edges are global per org and accumulate chunk_ids across
    # every ingestion that ever asserted that relationship (see
    # Neo4jDocumentGraphStore.upsert), unlike MENTIONS edges, which are
    # scoped to one specific, already-validated Version node. So a
    # RELATED edge's chunk_ids can span several different document
    # versions with different validity windows, and each one's
    # contributing version has to be checked individually via the
    # parallel chunk_document_version_ids array, not just the final
    # entity's own most recent mention.
    return f"""
    MATCH (anchor:Entity)
    WHERE anchor.org_id = $org_id
        AND toLower(anchor.name) IN $entities
    MATCH path = (anchor)-[:RELATED*0..{max_hops}]-(related:Entity)
    MATCH (version:Version)-[mentions:MENTIONS]->(related)
    WHERE version.org_id = $org_id
        AND version.valid_from <= datetime()
        AND (version.valid_to IS NULL OR version.valid_to > datetime())
    WITH anchor, related, version, path, mentions,
        reduce(
            ids = coalesce(mentions.chunk_ids, []),
            rel IN relationships(path) |
            ids + [
                i IN range(0, size(coalesce(rel.chunk_ids, [])) - 1)
                WHERE EXISTS {{
                    MATCH (rv:Version {{
                        org_id: $org_id,
                        document_version_id: rel.chunk_document_version_ids[i]
                    }})
                    WHERE rv.valid_from <= datetime()
                        AND (rv.valid_to IS NULL OR rv.valid_to > datetime())
                }}
                | rel.chunk_ids[i]
            ]
        ) AS chunk_ids,
        [rel IN relationships(path) | rel.kind] AS relation_kinds
    WITH anchor, related, version, chunk_ids, relation_kinds, length(path) AS hops
    WHERE size(chunk_ids) > 0
    RETURN DISTINCT
        version.document_id AS document_id,
        anchor.name AS anchor_entity,
        related.name AS related_entity,
        relation_kinds AS relation_kinds,
        chunk_ids AS chunk_ids,
        hops AS hops
    ORDER BY hops ASC
    LIMIT $limit
    """


def _evidence_from_row(row: Mapping[str, object]) -> GraphPathEvidence:
    return GraphPathEvidence(
        document_id=str(row["document_id"]),
        chunk_ids=_dedupe(str(chunk_id) for chunk_id in row["chunk_ids"]),
        anchor_entity=str(row["anchor_entity"]),
        related_entity=str(row["related_entity"]),
        relation_kinds=tuple(str(kind) for kind in row["relation_kinds"]),
        hops=int(row["hops"]),
    )


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)
