from __future__ import annotations

from src.modules.ingestion.domain.graph import DocumentGraph
from src.modules.ingestion.domain.models import DocumentVersionSource
from src.shared.app.ports import IGraphDb


class Neo4jDocumentGraphStore:
    def __init__(self, graphdb: IGraphDb) -> None:
        self._graphdb = graphdb

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
            SET edge.document_version_id = $document_version_id,
                edge.chunk_ids = reduce(
                    acc = coalesce(edge.chunk_ids, []),
                    chunk_id IN relation.chunk_ids |
                    CASE WHEN chunk_id IN acc THEN acc ELSE acc + chunk_id END
                )
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
