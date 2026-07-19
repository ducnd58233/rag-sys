from __future__ import annotations

import time
from collections.abc import Mapping

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from opentelemetry import propagate, trace

from src.shared.app.ports.message_queue import DlqTopic, MessageQueueError, Topic
from src.shared.configs.settings import KafkaSettings
from src.shared.observability.metrics import messaging_publish_duration

_tracer = trace.get_tracer(__name__)


class AioKafkaPublisher:
    def __init__(self, settings: KafkaSettings) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.bootstrap_servers,
            client_id=settings.client_id,
            acks="all",
            enable_idempotence=True,
        )

    async def start(self) -> None:
        try:
            await self._producer.start()
        except KafkaError as error:
            raise MessageQueueError("Could not start Kafka producer") from error

    async def publish(
        self,
        topic: Topic | DlqTopic,
        *,
        key: str | None,
        payload: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        with _tracer.start_as_current_span(
            f"{topic.value} publish",
            attributes={
                "messaging.system": "kafka",
                "messaging.destination.name": topic.value,
                "messaging.operation": "publish",
            },
        ):
            carrier: dict[str, str] = dict(headers or {})
            propagate.inject(carrier)
            encoded_headers = [
                (header_key, header_value.encode("utf-8"))
                for header_key, header_value in carrier.items()
            ]

            started_at = time.perf_counter()
            outcome = "success"
            try:
                await self._producer.send_and_wait(
                    topic.value,
                    value=payload,
                    key=key.encode("utf-8") if key is not None else None,
                    headers=encoded_headers,
                )
            except KafkaError as error:
                outcome = "failure"
                raise MessageQueueError(
                    f"Could not publish to {topic.value}"
                ) from error
            finally:
                messaging_publish_duration.record(
                    time.perf_counter() - started_at,
                    {"messaging.destination.name": topic.value, "outcome": outcome},
                )

    async def close(self) -> None:
        await self._producer.stop()
