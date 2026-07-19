from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.shared.app.mq.runtime import ConsumerRuntime
from src.shared.app.ports.message_queue import IMessageConsumer, IncomingMessage


class FakeConsumer(IMessageConsumer):
    def __init__(self) -> None:
        self.committed: list[IncomingMessage] = []

    async def start(self) -> None:
        return None

    async def poll(self, *, timeout_ms: int) -> Sequence[IncomingMessage]:
        return []

    async def commit(self, message: IncomingMessage) -> None:
        self.committed.append(message)

    async def stop(self) -> None:
        return None


class FakeGauge:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, str]]] = []

    def set(self, value: int, attributes: dict[str, str]) -> None:
        self.calls.append((value, attributes))


class FakeHistogram:
    def __init__(self) -> None:
        self.calls: list[tuple[float, dict[str, str]]] = []

    def record(self, value: float, attributes: dict[str, str]) -> None:
        self.calls.append((value, attributes))


class FakeCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, str]]] = []

    def add(self, value: int, attributes: dict[str, str]) -> None:
        self.calls.append((value, attributes))


@pytest.fixture
def message() -> IncomingMessage:
    return IncomingMessage(
        topic="document-ingestion-requested",
        partition=2,
        offset=10,
        key="document-version-1",
        payload=b"{}",
        headers={},
    )


@pytest.mark.asyncio
async def test_process_records_success_runtime_metrics(
    monkeypatch: pytest.MonkeyPatch,
    message: IncomingMessage,
) -> None:
    gauge = FakeGauge()
    histogram = FakeHistogram()
    counter = FakeCounter()
    monkeypatch.setattr("src.shared.app.mq.runtime.messaging_consume_inflight", gauge)
    monkeypatch.setattr("src.shared.app.mq.runtime.messaging_consume_duration", histogram)
    monkeypatch.setattr("src.shared.app.mq.runtime.messaging_commit_count", counter)
    consumer = FakeConsumer()

    async def handler(_: IncomingMessage) -> None:
        return None

    runtime = ConsumerRuntime(consumer=consumer, handler=handler)

    await runtime._process(message)

    assert gauge.calls == [
        (1, {"messaging.destination.name": message.topic, "partition": "2"}),
        (0, {"messaging.destination.name": message.topic, "partition": "2"}),
    ]
    assert histogram.calls
    assert histogram.calls[0][1] == {
        "messaging.destination.name": message.topic,
        "outcome": "success",
    }
    assert consumer.committed == [message]
    assert counter.calls == [
        (1, {"messaging.destination.name": message.topic, "partition": "2"}),
    ]


@pytest.mark.asyncio
async def test_process_records_failure_runtime_metrics_without_commit(
    monkeypatch: pytest.MonkeyPatch,
    message: IncomingMessage,
) -> None:
    gauge = FakeGauge()
    histogram = FakeHistogram()
    counter = FakeCounter()
    monkeypatch.setattr("src.shared.app.mq.runtime.messaging_consume_inflight", gauge)
    monkeypatch.setattr("src.shared.app.mq.runtime.messaging_consume_duration", histogram)
    monkeypatch.setattr("src.shared.app.mq.runtime.messaging_commit_count", counter)
    consumer = FakeConsumer()

    async def handler(_: IncomingMessage) -> None:
        raise RuntimeError("boom")

    runtime = ConsumerRuntime(consumer=consumer, handler=handler)

    await runtime._process(message)

    assert gauge.calls == [
        (1, {"messaging.destination.name": message.topic, "partition": "2"}),
        (0, {"messaging.destination.name": message.topic, "partition": "2"}),
    ]
    assert histogram.calls
    assert histogram.calls[0][1] == {
        "messaging.destination.name": message.topic,
        "outcome": "failure",
    }
    assert consumer.committed == []
    assert counter.calls == []
