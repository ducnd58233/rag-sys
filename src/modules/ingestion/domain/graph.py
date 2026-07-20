from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphEntity:
    name: str
    kind: str
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphRelation:
    source: str
    target: str
    kind: str
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentGraph:
    entities: tuple[GraphEntity, ...]
    relations: tuple[GraphRelation, ...]
