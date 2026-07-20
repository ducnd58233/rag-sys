from __future__ import annotations

from collections.abc import Sequence

from src.shared.app.ports import IGraphDb


class Neo4jGraphSearcher:
    def __init__(self, graphdb: IGraphDb) -> None:
        self._graphdb = graphdb

    async def related_document_ids(
        self,
        *,
        org_id: int,
        entities: Sequence[str],
        max_hops: int,
        limit: int,
    ) -> Sequence[str]:
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
        return tuple(str(row["document_id"]) for row in rows)


def _query(max_hops: int) -> str:
    return f"""
    MATCH (anchor:Entity)
    WHERE anchor.org_id = $org_id
        AND toLower(anchor.name) IN $entities
    MATCH (anchor)-[:RELATED*0..{max_hops}]-(related:Entity)
    MATCH (version:Version)-[:MENTIONS]->(related)
    WHERE version.org_id = $org_id
        AND version.valid_from <= datetime()
        AND (version.valid_to IS NULL OR version.valid_to > datetime())
    RETURN DISTINCT version.document_id AS document_id
    LIMIT $limit
    """
