from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from opentelemetry import propagate, trace

from src.shared.app.mq.retry import RetryPolicy, is_retryable
from src.shared.app.ports.message_queue import (
    DlqTopic,
    IMessageConsumer,
    IMessagePublisher,
    IncomingMessage,
    MessageHeader,
    Topic,
)
from src.shared.kernel.errors import DomainException
from src.shared.observability.metrics import (
    messaging_commit_count,
    messaging_consume_duration,
    messaging_consume_inflight,
    messaging_dlq_count,
)

logger = logging.getLogger(__name__)

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]

_tracer = trace.get_tracer(__name__)


@dataclass(frozen=True, slots=True)
class ConsumerRuntimeConfig:
    topic: Topic
    poll_timeout_ms: int = 1_000
    retry: RetryPolicy = RetryPolicy()


class ConsumerRuntime:
    def __init__(
        self,
        *,
        consumer: IMessageConsumer,
        publisher: IMessagePublisher,
        handler: MessageHandler,
        config: ConsumerRuntimeConfig,
    ) -> None:
        self._consumer = consumer
        self._publisher = publisher
        self._handler = handler
        self._config = config
        self._dlq_topic = DlqTopic.for_topic(config.topic)
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        await self._consumer.start()
        try:
            while not self._stopping.is_set():
                messages = await self._consumer.poll(
                    timeout_ms=self._config.poll_timeout_ms,
                )
                for message in messages:
                    await self._process(message)
        finally:
            await self._consumer.stop()

    def request_stop(self) -> None:
        self._stopping.set()

    async def _process(self, message: IncomingMessage) -> None:
        context = propagate.extract(message.headers)
        metric_attributes = {
            "messaging.destination.name": message.topic,
            "partition": str(message.partition),
        }
        messaging_consume_inflight.set(1, metric_attributes)
        try:
            with _tracer.start_as_current_span(
                f"{message.topic} process",
                context=context,
                attributes={
                    "messaging.system": "kafka",
                    "messaging.destination.name": message.topic,
                    "messaging.operation": "process",
                },
            ):
                error, attempts = await self._handle_with_retries(message)
        finally:
            messaging_consume_inflight.set(0, metric_attributes)

        if error is None:
            await self._consumer.commit(message)
            messaging_commit_count.add(1, metric_attributes)
            return

        # FR-ACK-2: commit only once the failure is durably recorded in the DLQ.
        try:
            await self._publish_to_dlq(message, error, attempts)
        except Exception:
            logger.exception(
                "Could not write message to DLQ; leaving offset uncommitted",
                extra=self._log_context(message) | {"attempt": attempts},
            )
            return

        messaging_dlq_count.add(1, metric_attributes)
        await self._consumer.commit(message)
        messaging_commit_count.add(1, metric_attributes)

    async def _handle_with_retries(
        self,
        message: IncomingMessage,
    ) -> tuple[Exception | None, int]:
        attempt = 0
        while True:
            attempt += 1
            started_at = time.perf_counter()
            outcome = "success"
            try:
                await self._handler(message)
                return None, attempt
            except Exception as error:
                outcome = "failure"
                if (
                    not is_retryable(error)
                    or attempt >= self._config.retry.max_attempts
                ):
                    logger.warning(
                        "Message handling failed permanently",
                        extra=self._log_context(message) | {"attempt": attempt},
                    )
                    return error, attempt

                delay = self._config.retry.delay_for(attempt)
                logger.warning(
                    "Message handling failed; retrying",
                    extra=self._log_context(message)
                    | {"attempt": attempt, "delay_seconds": delay},
                )
                await asyncio.sleep(delay)
            finally:
                messaging_consume_duration.record(
                    time.perf_counter() - started_at,
                    {
                        "messaging.destination.name": message.topic,
                        "outcome": outcome,
                    },
                )

    async def _publish_to_dlq(
        self,
        message: IncomingMessage,
        error: Exception,
        attempts: int,
    ) -> None:
        error_code = (
            error.code.value if isinstance(error, DomainException) else "UNKNOWN"
        )
        headers = {
            MessageHeader.ORIGINAL_TOPIC.value: message.topic,
            MessageHeader.ORIGINAL_PARTITION.value: str(message.partition),
            MessageHeader.ORIGINAL_OFFSET.value: str(message.offset),
            MessageHeader.ERROR_CODE.value: error_code,
            MessageHeader.ERROR_MESSAGE.value: str(error)[:1000],
            MessageHeader.ATTEMPTS.value: str(attempts),
            MessageHeader.FAILED_AT.value: datetime.now(UTC).isoformat(),
        }
        await self._publisher.publish(
            self._dlq_topic,
            key=message.key,
            payload=message.payload,
            headers=headers,
        )

    def _log_context(self, message: IncomingMessage) -> dict[str, object]:
        return {
            "topic": message.topic,
            "partition": message.partition,
            "offset": message.offset,
        }
