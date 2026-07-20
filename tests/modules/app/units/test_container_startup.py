from __future__ import annotations

import pytest

from src.bootstrap import AppContainer


class FakeVectorStore:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def create_index_if_not_exists(self) -> None:
        self._events.append("index")


class FakeKafkaPublisher:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def start(self) -> None:
        self._events.append("kafka")


@pytest.mark.asyncio
async def test_startup_prepares_vector_index_before_kafka() -> None:
    events: list[str] = []
    container = AppContainer(
        settings=object(),
        observability=object(),
        database=object(),
        object_storage=object(),
        elasticsearch=object(),
        embedding_model=object(),
        kafka_publisher=FakeKafkaPublisher(events),
        create_document_upload=object(),
        complete_document_upload=object(),
        publish_ingestion_outbox_events=object(),
        vector_store=FakeVectorStore(events),
        ingest_document=object(),
        request_ingestion=object(),
        ingestion_handler=object(),
        retrieve=object(),
        answer_question=object(),
    )

    await container.startup()

    assert events == ["index", "kafka"]
