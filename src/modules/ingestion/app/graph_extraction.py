from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from src.modules.ingestion.domain.graph import (
    DocumentGraph,
    GraphEntity,
    GraphRelation,
)
from src.modules.ingestion.domain.models import Chunk, DocumentVersionSource
from src.shared.app.ports import IChatModel

_SYSTEM = """
Extract graph facts for retrieval.
Return JSON with entities and directed relations stated or strongly implied by the text.
Entity names must be short canonical noun phrases.
Relation kinds must be lowercase snake_case.
Do not invent facts that are not supported by the text.
""".strip()


class GraphEntitySchema(BaseModel):
    name: str = Field(min_length=1)
    kind: str = Field(default="concept", min_length=1)


class GraphRelationSchema(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: str = Field(min_length=1)


class DocumentGraphSchema(BaseModel):
    entities: list[GraphEntitySchema] = Field(default_factory=list)
    relations: list[GraphRelationSchema] = Field(default_factory=list)


class LlmDocumentGraphExtractor:
    def __init__(
        self,
        chat_model: IChatModel,
        max_characters: int,
    ) -> None:
        self._chat = chat_model
        self._max_characters = max_characters

    async def extract(
        self,
        *,
        source: DocumentVersionSource,
        chunks: Sequence[Chunk],
    ) -> DocumentGraph:
        if not chunks:
            return DocumentGraph(entities=(), relations=())
        result = await self._chat.complete_structured(
            system=_SYSTEM,
            user=_prompt(source, chunks, self._max_characters),
            schema=DocumentGraphSchema,
            temperature=0.0,
            max_tokens=1024,
        )
        entities = {
            entity.name.casefold(): GraphEntity(
                name=entity.name.strip(),
                kind=entity.kind.strip().casefold(),
            )
            for entity in result.entities
            if entity.name.strip()
        }
        relations = tuple(
            GraphRelation(
                source=relation.source.strip(),
                target=relation.target.strip(),
                kind=relation.kind.strip().casefold(),
            )
            for relation in result.relations
            if relation.source.strip() and relation.target.strip()
        )
        for relation in relations:
            entities.setdefault(
                relation.source.casefold(),
                GraphEntity(name=relation.source, kind="concept"),
            )
            entities.setdefault(
                relation.target.casefold(),
                GraphEntity(name=relation.target, kind="concept"),
            )
        return DocumentGraph(
            entities=tuple(entities.values()),
            relations=relations,
        )


def _prompt(
    source: DocumentVersionSource,
    chunks: Sequence[Chunk],
    max_characters: int,
) -> str:
    text = "\n\n".join(chunk.content for chunk in chunks)
    return (
        f"DOCUMENT_ID: {source.document_id.value}\n"
        f"DOCUMENT_VERSION_ID: {source.document_version_id}\n"
        f"FILENAME: {source.filename}\n"
        f"TEXT:\n{text[:max_characters]}"
    )
