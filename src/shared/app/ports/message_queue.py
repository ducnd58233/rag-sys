from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class MessageQueueError(RuntimeError):
    pass


class Topic(StrEnum):
    DOCUMENT_INGESTION_REQUESTED = "document.ingestion.requested.v1"


class ConsumerGroup(StrEnum):
    INGESTION_WORKER = "ingestion-worker"


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    topic: str
    partition: int
    offset: int
    key: str | None
    payload: bytes


class IMessagePublisher(Protocol):
    async def publish(
        self,
        topic: Topic,
        *,
        key: str | None,
        payload: bytes,
    ) -> None: ...


class IMessageConsumer(Protocol):
    async def start(self) -> None: ...

    async def poll(self, *, timeout_ms: int) -> Sequence[IncomingMessage]: ...

    async def commit(self, message: IncomingMessage) -> None: ...

    async def stop(self) -> None: ...
