from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from src.modules.ingestion.domain.graph import DocumentGraph, GraphEntity, GraphRelation
from src.modules.ingestion.domain.models import Chunk, DocumentVersionSource
from src.shared.app.ports import IChatModel

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """
Extract graph facts for retrieval from one document chunk.
Return JSON with entities and directed relations stated or strongly implied by
the text in this chunk only.
Entity names must be short canonical noun phrases.
Relation kinds must be lowercase snake_case.
Do not invent facts that are not supported by the text.
""".strip()

_CANONICALIZATION_SYSTEM = """
Group entity name aliases that refer to the same real-world entity.
Only group names that are clearly the same entity written differently, such as
an abbreviation, casing, punctuation, or singular/plural variation. Do not
group names that are merely related or similar in topic.
Return JSON with:
- groups: list of {canonical_name, aliases}
Every input name that has at least one alias must appear in exactly one
group, either as canonical_name or in aliases. Names with no alias must not
appear in any group.
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


class CanonicalGroupSchema(BaseModel):
    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)


class EntityCanonicalizationSchema(BaseModel):
    groups: list[CanonicalGroupSchema] = Field(default_factory=list)


class LlmDocumentGraphExtractor:
    def __init__(
        self,
        chat_model: IChatModel,
        *,
        extraction_max_characters: int,
        extraction_max_tokens: int = 1024,
        canonicalization_max_tokens: int = 512,
    ) -> None:
        self._chat = chat_model
        self._extraction_max_characters = extraction_max_characters
        self._extraction_max_tokens = extraction_max_tokens
        self._canonicalization_max_tokens = canonicalization_max_tokens

    async def extract(
        self,
        *,
        source: DocumentVersionSource,
        chunks: Sequence[Chunk],
    ) -> DocumentGraph:
        if not chunks:
            return DocumentGraph(entities=(), relations=())

        entities: dict[str, GraphEntity] = {}
        relations: dict[tuple[str, str, str], GraphRelation] = {}
        for chunk in chunks:
            result = await self._extract_chunk(source, chunk)
            _merge_entities(entities, result.entities, chunk.chunk_id)
            _merge_relations(relations, result.relations, chunk.chunk_id)

        if not entities:
            return DocumentGraph(entities=(), relations=())

        canonical_map = await self._canonicalize(
            entity.name for entity in entities.values()
        )
        if canonical_map:
            entities = _apply_canonical_entities(entities, canonical_map)
            relations = _apply_canonical_relations(relations, canonical_map)

        return DocumentGraph(
            entities=tuple(entities.values()),
            relations=tuple(relations.values()),
        )

    async def _extract_chunk(
        self,
        source: DocumentVersionSource,
        chunk: Chunk,
    ) -> DocumentGraphSchema:
        # Graph enrichment is best-effort: a bad chunk must not fail the whole document.
        try:
            return await self._chat.complete_structured(
                system=_EXTRACTION_SYSTEM,
                user=_chunk_prompt(source, chunk, self._extraction_max_characters),
                schema=DocumentGraphSchema,
                temperature=0.0,
                max_tokens=self._extraction_max_tokens,
            )
        except Exception:
            logger.exception(
                "graph extraction failed for chunk; skipping graph facts for it",
                extra={"chunk_id": chunk.chunk_id},
            )
            return DocumentGraphSchema()

    async def _canonicalize(self, names: Sequence[str]) -> dict[str, str]:
        distinct = sorted({name.strip() for name in names if name.strip()})
        if len(distinct) < 2:
            return {}
        try:
            result = await self._chat.complete_structured(
                system=_CANONICALIZATION_SYSTEM,
                user="NAMES:\n" + "\n".join(distinct),
                schema=EntityCanonicalizationSchema,
                temperature=0.0,
                max_tokens=self._canonicalization_max_tokens,
            )
        except Exception:
            logger.exception("entity canonicalization failed; keeping raw entity names")
            return {}
        return _canonical_map_from_groups(result.groups)


def _chunk_prompt(
    source: DocumentVersionSource,
    chunk: Chunk,
    max_characters: int,
) -> str:
    return (
        f"DOCUMENT_ID: {source.document_id.value}\n"
        f"DOCUMENT_VERSION_ID: {source.document_version_id}\n"
        f"FILENAME: {source.filename}\n"
        f"CHUNK_ID: {chunk.chunk_id}\n"
        f"TEXT:\n{chunk.content[:max_characters]}"
    )


def _merge_entities(
    entities: dict[str, GraphEntity],
    raw_entities: Sequence[GraphEntitySchema],
    chunk_id: str,
) -> None:
    for raw in raw_entities:
        name = raw.name.strip()
        if not name:
            continue
        key = name.casefold()
        existing = entities.get(key)
        if existing is None:
            entities[key] = GraphEntity(
                name=name,
                kind=raw.kind.strip().casefold() or "concept",
                chunk_ids=(chunk_id,),
            )
            continue
        entities[key] = GraphEntity(
            name=existing.name,
            kind=existing.kind,
            chunk_ids=_add_chunk_id(existing.chunk_ids, chunk_id),
        )


def _merge_relations(
    relations: dict[tuple[str, str, str], GraphRelation],
    raw_relations: Sequence[GraphRelationSchema],
    chunk_id: str,
) -> None:
    for raw in raw_relations:
        source = raw.source.strip()
        target = raw.target.strip()
        kind = raw.kind.strip().casefold()
        if not source or not target or not kind:
            continue
        key = (source.casefold(), target.casefold(), kind)
        existing = relations.get(key)
        if existing is None:
            relations[key] = GraphRelation(
                source=source,
                target=target,
                kind=kind,
                chunk_ids=(chunk_id,),
            )
            continue
        relations[key] = GraphRelation(
            source=existing.source,
            target=existing.target,
            kind=existing.kind,
            chunk_ids=_add_chunk_id(existing.chunk_ids, chunk_id),
        )


def _add_chunk_id(chunk_ids: tuple[str, ...], chunk_id: str) -> tuple[str, ...]:
    return chunk_ids if chunk_id in chunk_ids else (*chunk_ids, chunk_id)


def _union_chunk_ids(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    merged = list(left)
    for chunk_id in right:
        if chunk_id not in merged:
            merged.append(chunk_id)
    return tuple(merged)


def _canonical_map_from_groups(
    groups: Sequence[CanonicalGroupSchema],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group in groups:
        canonical_name = group.canonical_name.strip()
        if not canonical_name:
            continue
        mapping[canonical_name.casefold()] = canonical_name
        for alias in group.aliases:
            alias = alias.strip()
            if alias:
                mapping[alias.casefold()] = canonical_name
    return mapping


def _resolve_canonical_name(name: str, canonical_map: Mapping[str, str]) -> str:
    return canonical_map.get(name.casefold(), name)


def _apply_canonical_entities(
    entities: Mapping[str, GraphEntity],
    canonical_map: Mapping[str, str],
) -> dict[str, GraphEntity]:
    resolved: dict[str, GraphEntity] = {}
    for entity in entities.values():
        canonical_name = _resolve_canonical_name(entity.name, canonical_map)
        key = canonical_name.casefold()
        existing = resolved.get(key)
        if existing is None:
            resolved[key] = GraphEntity(
                name=canonical_name,
                kind=entity.kind,
                chunk_ids=entity.chunk_ids,
            )
            continue
        resolved[key] = GraphEntity(
            name=existing.name,
            kind=existing.kind,
            chunk_ids=_union_chunk_ids(existing.chunk_ids, entity.chunk_ids),
        )
    return resolved


def _apply_canonical_relations(
    relations: Mapping[tuple[str, str, str], GraphRelation],
    canonical_map: Mapping[str, str],
) -> dict[tuple[str, str, str], GraphRelation]:
    resolved: dict[tuple[str, str, str], GraphRelation] = {}
    for relation in relations.values():
        source = _resolve_canonical_name(relation.source, canonical_map)
        target = _resolve_canonical_name(relation.target, canonical_map)
        key = (source.casefold(), target.casefold(), relation.kind)
        existing = resolved.get(key)
        if existing is None:
            resolved[key] = GraphRelation(
                source=source,
                target=target,
                kind=relation.kind,
                chunk_ids=relation.chunk_ids,
            )
            continue
        resolved[key] = GraphRelation(
            source=existing.source,
            target=existing.target,
            kind=existing.kind,
            chunk_ids=_union_chunk_ids(existing.chunk_ids, relation.chunk_ids),
        )
    return resolved
