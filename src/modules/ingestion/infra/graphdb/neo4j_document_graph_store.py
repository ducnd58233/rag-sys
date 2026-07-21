from __future__ import annotations

from datetime import datetime

from src.modules.ingestion.domain.graph import DocumentGraph
from src.modules.ingestion.domain.models import DocumentId, DocumentVersionSource
from src.shared.app.ports import IGraphDb


class Neo4jDocumentGraphStore:
    def __init__(self, graphdb: IGraphDb) -> None:
        self._graphdb = graphdb

    async def close_superseded_versions(
        self,
        *,
        org_id: int,
        document_id: DocumentId,
        active_document_version_id: int,
        valid_to: datetime,
    ) -> None:
        await self._graphdb.write(
            """
            MATCH (v:Version {org_id: $org_id, document_id: $document_id})
            WHERE v.document_version_id <> $active_document_version_id
                AND v.valid_from < datetime($valid_to)
                AND v.valid_to IS NULL
            SET v.valid_to = datetime($valid_to)
            """,
            {
                "org_id": org_id,
                "document_id": document_id.value,
                "active_document_version_id": active_document_version_id,
                "valid_to": valid_to.isoformat(),
            },
        )

    async def upsert(
        self,
        *,
        source: DocumentVersionSource,
        graph: DocumentGraph,
    ) -> None:
        await self._graphdb.write(
            """
            MERGE (d:Document {org_id: $org_id, document_id: $document_id})
            SET d.filename = $filename
            MERGE (v:Version {
                org_id: $org_id,
                document_version_id: $document_version_id
            })
            SET v.document_id = $document_id,
                v.version_no = $version_no,
                v.valid_from = datetime($valid_from),
                v.valid_to = CASE
                    WHEN $valid_to IS NULL THEN NULL
                    ELSE datetime($valid_to)
                END
            MERGE (d)-[:HAS_VERSION]->(v)
            WITH v
            UNWIND $entities AS entity
            MERGE (e:Entity {org_id: $org_id, name: entity.name})
            SET e.kind = entity.kind
            MERGE (v)-[mentions:MENTIONS]->(e)
            SET mentions.chunk_ids = entity.chunk_ids
            """,
            {
                "org_id": source.org_id,
                "document_id": source.document_id.value,
                "document_version_id": source.document_version_id,
                "version_no": source.version_no,
                "filename": source.filename,
                "valid_from": source.valid_from.isoformat(),
                "valid_to": (
                    source.valid_to.isoformat() if source.valid_to is not None else None
                ),
                "entities": [
                    {
                        "name": entity.name,
                        "kind": entity.kind,
                        "chunk_ids": list(entity.chunk_ids),
                    }
                    for entity in graph.entities
                ],
            },
        )
        await self._graphdb.write(
            """
            MATCH (v:Version {
                org_id: $org_id,
                document_version_id: $document_version_id
            })
            UNWIND $relations AS relation
            MATCH (source:Entity {org_id: $org_id, name: relation.source})
            MATCH (target:Entity {org_id: $org_id, name: relation.target})
            MERGE (source)-[edge:RELATED {kind: relation.kind}]->(target)
            WITH edge, relation, reduce(
                acc = {
                    chunk_ids: coalesce(edge.chunk_ids, []),
                    chunk_document_version_ids: coalesce(edge.chunk_document_version_ids, [])
                },
                chunk_id IN relation.chunk_ids |
                CASE WHEN chunk_id IN acc.chunk_ids
                    THEN acc
                    ELSE {
                        chunk_ids: acc.chunk_ids + chunk_id,
                        chunk_document_version_ids: acc.chunk_document_version_ids + $document_version_id
                    }
                END
            ) AS merged
            SET edge.document_version_id = $document_version_id,
                edge.chunk_ids = merged.chunk_ids,
                edge.chunk_document_version_ids = merged.chunk_document_version_ids
            """,
            {
                "org_id": source.org_id,
                "document_version_id": source.document_version_id,
                "relations": [
                    {
                        "source": relation.source,
                        "target": relation.target,
                        "kind": relation.kind,
                        "chunk_ids": list(relation.chunk_ids),
                    }
                    for relation in graph.relations
                ],
            },
        )
        if source.version_no > 1:
            await self._graphdb.write(
                """
                MATCH (current:Version {
                    org_id: $org_id,
                    document_id: $document_id,
                    version_no: $version_no
                })
                MATCH (previous:Version {
                    org_id: $org_id,
                    document_id: $document_id,
                    version_no: $previous_version_no
                })
                MERGE (current)-[:SUPERSEDES]->(previous)
                """,
                {
                    "org_id": source.org_id,
                    "document_id": source.document_id.value,
                    "version_no": source.version_no,
                    "previous_version_no": source.version_no - 1,
                },
            )
