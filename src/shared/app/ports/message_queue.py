from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class MessageQueueError(RuntimeError):
    pass


DLQ_TOPIC_SUFFIX = ".dlq"


class Topic(StrEnum):
    DOCUMENT_INGESTION_REQUESTED = "document.ingestion.requested.v1"


class DlqTopic(StrEnum):
    DOCUMENT_INGESTION_REQUESTED = "document.ingestion.requested.v1.dlq"

    @classmethod
    def for_topic(cls, topic: Topic) -> DlqTopic:
        return cls[topic.name]


class ConsumerGroup(StrEnum):
    INGESTION_WORKER = "ingestion-worker"


class MessageHeader(StrEnum):
    ORIGINAL_TOPIC = "x-original-topic"
    ORIGINAL_PARTITION = "x-original-partition"
    ORIGINAL_OFFSET = "x-original-offset"
    ERROR_CODE = "x-error-code"
    ERROR_MESSAGE = "x-error-message"
    ATTEMPTS = "x-attempts"
    FAILED_AT = "x-failed-at"


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    topic: str
    partition: int
    offset: int
    key: str | None
    payload: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class IMessagePublisher(Protocol):
    async def publish(
        self,
        topic: Topic | DlqTopic,
        *,
        key: str | None,
        payload: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> None: ...


class IMessageConsumer(Protocol):
    async def start(self) -> None: ...

    async def poll(self, *, timeout_ms: int) -> Sequence[IncomingMessage]: ...

    async def commit(self, message: IncomingMessage) -> None: ...

    async def stop(self) -> None: ...
