from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from src.modules.document.domain.models import DocumentProcessingStatus
from src.modules.ingestion.app.graph_extraction import LlmDocumentGraphExtractor
from src.modules.ingestion.domain.models import Chunk, DocumentId, DocumentVersionSource


class FakeChatModel:
    def __init__(self, results: Sequence[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._results = list(results)

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return schema.model_validate(result)

    async def complete(self, **kwargs: object):
        raise AssertionError("complete should not be called")

    def bind_tools(self, tools: Sequence[object]):
        return self


def _chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=DocumentId(42),
        index=0,
        content=content,
        metadata={},
    )


def _source() -> DocumentVersionSource:
    return DocumentVersionSource(
        org_id=7,
        document_id=DocumentId(42),
        document_version_id=101,
        storage_object_id=201,
        version_no=1,
        bucket="documents",
        object_key="paper.pdf",
        filename="paper.pdf",
        mime_type="application/pdf",
        processing_status=DocumentProcessingStatus.PARSING,
        size_bytes=1000,
        checksum_sha256=None,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_to=None,
    )


@pytest.mark.asyncio
async def test_extract_returns_empty_graph_without_any_call_for_no_chunks() -> None:
    chat = FakeChatModel([])
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)

    graph = await extractor.extract(source=_source(), chunks=())

    assert graph.entities == ()
    assert graph.relations == ()
    assert chat.calls == []


@pytest.mark.asyncio
async def test_extract_runs_one_llm_call_per_chunk_and_tags_chunk_ids() -> None:
    chat = FakeChatModel(
        [
            {
                "entities": [{"name": "Checkout", "kind": "service"}],
                "relations": [],
            },
            {
                "entities": [{"name": "Payment", "kind": "service"}],
                "relations": [
                    {"source": "Checkout", "target": "Payment", "kind": "depends_on"}
                ],
            },
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)
    chunks = (
        _chunk("42:1:0", "Checkout service overview."),
        _chunk("42:1:1", "Checkout depends on Payment."),
    )

    graph = await extractor.extract(source=_source(), chunks=chunks)

    entities_by_name = {entity.name: entity for entity in graph.entities}
    assert entities_by_name["Checkout"].chunk_ids == ("42:1:0",)
    assert entities_by_name["Payment"].chunk_ids == ("42:1:1",)
    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert (relation.source, relation.target, relation.kind) == (
        "Checkout",
        "Payment",
        "depends_on",
    )
    assert relation.chunk_ids == ("42:1:1",)
    assert len(chat.calls) == 3
    assert chat.calls[0]["user"].count("CHUNK_ID: 42:1:0") == 1
    assert chat.calls[1]["user"].count("CHUNK_ID: 42:1:1") == 1


@pytest.mark.asyncio
async def test_extract_merges_same_entity_seen_in_multiple_chunks() -> None:
    chat = FakeChatModel(
        [
            {"entities": [{"name": "Checkout", "kind": "service"}], "relations": []},
            {"entities": [{"name": "checkout", "kind": "service"}], "relations": []},
            {"groups": []},
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)
    chunks = (
        _chunk("42:1:0", "Checkout overview."),
        _chunk("42:1:1", "More on checkout."),
    )

    graph = await extractor.extract(source=_source(), chunks=chunks)

    assert len(graph.entities) == 1
    assert graph.entities[0].chunk_ids == ("42:1:0", "42:1:1")


@pytest.mark.asyncio
async def test_extraction_failure_for_one_chunk_is_skipped_not_raised() -> None:
    chat = FakeChatModel(
        [
            RuntimeError("output parsing failure"),
            {
                "entities": [{"name": "Payment", "kind": "service"}],
                "relations": [],
            },
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)
    chunks = (
        _chunk("42:1:0", "Broken chunk."),
        _chunk("42:1:1", "Payment service overview."),
    )

    graph = await extractor.extract(source=_source(), chunks=chunks)

    assert len(graph.entities) == 1
    assert graph.entities[0].name == "Payment"
    assert graph.relations == ()


@pytest.mark.asyncio
async def test_canonicalization_merges_aliases_across_chunks() -> None:
    chat = FakeChatModel(
        [
            {"entities": [{"name": "DB", "kind": "component"}], "relations": []},
            {
                "entities": [{"name": "Database", "kind": "component"}],
                "relations": [],
            },
            {
                "groups": [
                    {"canonical_name": "Database", "aliases": ["DB"]},
                ]
            },
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)
    chunks = (
        _chunk("42:1:0", "The DB stores orders."),
        _chunk("42:1:1", "Database schema details."),
    )

    graph = await extractor.extract(source=_source(), chunks=chunks)

    assert len(graph.entities) == 1
    entity = graph.entities[0]
    assert entity.name == "Database"
    assert entity.chunk_ids == ("42:1:0", "42:1:1")
    assert len(chat.calls) == 3
    canonicalization_call = chat.calls[2]
    assert "DB" in canonicalization_call["user"]
    assert "Database" in canonicalization_call["user"]
    assert canonicalization_call["temperature"] == 0.0


@pytest.mark.asyncio
async def test_canonicalization_skipped_when_fewer_than_two_distinct_entities() -> None:
    chat = FakeChatModel(
        [
            {"entities": [{"name": "Checkout", "kind": "service"}], "relations": []},
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)

    graph = await extractor.extract(
        source=_source(),
        chunks=(_chunk("42:1:0", "Checkout overview."),),
    )

    assert len(graph.entities) == 1
    assert len(chat.calls) == 1


@pytest.mark.asyncio
async def test_canonicalization_failure_keeps_raw_extraction_result() -> None:
    chat = FakeChatModel(
        [
            {"entities": [{"name": "DB", "kind": "component"}], "relations": []},
            {
                "entities": [{"name": "Database", "kind": "component"}],
                "relations": [],
            },
            RuntimeError("canonicalization model unavailable"),
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)
    chunks = (
        _chunk("42:1:0", "The DB stores orders."),
        _chunk("42:1:1", "Database schema details."),
    )

    graph = await extractor.extract(source=_source(), chunks=chunks)

    names = sorted(entity.name for entity in graph.entities)
    assert names == ["DB", "Database"]


@pytest.mark.asyncio
async def test_relation_endpoints_are_remapped_through_canonical_names() -> None:
    chat = FakeChatModel(
        [
            {
                "entities": [{"name": "DB", "kind": "component"}],
                "relations": [],
            },
            {
                "entities": [{"name": "Checkout", "kind": "service"}],
                "relations": [
                    {"source": "Checkout", "target": "DB", "kind": "depends_on"}
                ],
            },
            {"groups": [{"canonical_name": "Database", "aliases": ["DB"]}]},
        ]
    )
    extractor = LlmDocumentGraphExtractor(chat, extraction_max_characters=1000)
    chunks = (
        _chunk("42:1:0", "The DB stores orders."),
        _chunk("42:1:1", "Checkout depends on the DB."),
    )

    graph = await extractor.extract(source=_source(), chunks=chunks)

    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.source == "Checkout"
    assert relation.target == "Database"
