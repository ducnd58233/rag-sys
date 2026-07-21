from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.bootstrap import AppContainer
from src.shared.app.ports import ChatResult
from src.shared.configs.settings import ResilienceSettings, Settings


class FakeVectorStore:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def create_index_if_not_exists(self) -> None:
        self._events.append("index")


class FakeGraphDb:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def initialize(self) -> None:
        self._events.append("graphdb")


class FakeKafkaPublisher:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def start(self) -> None:
        self._events.append("kafka")


class FakeEmbeddingModel:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    @property
    def dimensions(self) -> int:
        return 8

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._events.append(f"embed:{texts[0]}")
        return [[0.0] * self.dimensions]


class FakeChatModel:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> ChatResult:
        del system, temperature
        self._events.append(f"chat:{user}")
        return ChatResult(content="pong")

    async def complete_structured(self, **kwargs: object) -> object:
        raise NotImplementedError

    def bind_tools(self, tools: Sequence[object]) -> FakeChatModel:
        del tools
        return self


def _settings(*, warmup: ResilienceSettings) -> Settings:
    return Settings().model_copy(update={"resilience": warmup})


def _container(events: list[str], *, settings: Settings) -> AppContainer:
    return AppContainer(
        settings=settings,
        observability=object(),
        database=object(),
        object_storage=object(),
        elasticsearch=object(),
        graphdb=FakeGraphDb(events),
        embedding_model=FakeEmbeddingModel(events),
        chat_model=FakeChatModel(events),
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


@pytest.mark.asyncio
async def test_startup_warms_embedding_and_chat_when_enabled() -> None:
    events: list[str] = []
    settings = _settings(
        warmup=ResilienceSettings(
            warmup_enabled=True,
            warmup_embedding=True,
            warmup_chat=True,
            warmup_embedding_text="embed-ping",
            warmup_chat_user="chat-ping",
        )
    )

    await _container(events, settings=settings).startup()

    assert events == [
        "graphdb",
        "index",
        "kafka",
        "embed:embed-ping",
        "chat:chat-ping",
    ]


@pytest.mark.asyncio
async def test_startup_skips_warmup_when_disabled() -> None:
    events: list[str] = []
    settings = _settings(warmup=ResilienceSettings(warmup_enabled=False))

    await _container(events, settings=settings).startup()

    assert events == ["graphdb", "index", "kafka"]
