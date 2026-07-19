from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import TypeVar

from src.bootstrap import build_container
from src.shared.app.mq.retry import RetryPolicy
from src.shared.app.mq.runtime import ConsumerRuntime, ConsumerRuntimeConfig
from src.shared.app.ports.message_queue import ConsumerGroup, Topic
from src.shared.app.scheduler import PeriodicRuntime
from src.shared.infra.mq import AioKafkaConsumer

_T = TypeVar("_T")


def _discard_result(
    func: Callable[[], Awaitable[_T]],
) -> Callable[[], Awaitable[None]]:
    async def _call() -> None:
        await func()

    return _call


async def _run() -> None:
    container = build_container(service_name="worker")
    await container.startup()

    consumer = AioKafkaConsumer(
        container.settings.kafka,
        topic=Topic.DOCUMENT_INGESTION_REQUESTED,
        group=ConsumerGroup.INGESTION_WORKER,
    )
    consumer_runtime = ConsumerRuntime(
        consumer=consumer,
        publisher=container.kafka_publisher,
        handler=container.ingestion_handler.handle,
        config=ConsumerRuntimeConfig(
            topic=Topic.DOCUMENT_INGESTION_REQUESTED,
            poll_timeout_ms=container.settings.kafka.consumer_poll_timeout_ms,
            retry=RetryPolicy(
                max_attempts=container.settings.kafka.retry_max_attempts,
                base_delay_seconds=container.settings.kafka.retry_base_delay_seconds,
                max_delay_seconds=container.settings.kafka.retry_max_delay_seconds,
            ),
        ),
    )
    runtimes = [consumer_runtime]

    if container.settings.ingestion.outbox_relay_enabled:
        outbox_relay_runtime = PeriodicRuntime(
            task=_discard_result(container.publish_ingestion_outbox_events.execute),
            interval_seconds=container.settings.ingestion.outbox_relay_interval_seconds,
            name="publish-ingestion-outbox-events",
        )
        runtimes.append(outbox_relay_runtime)

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: [runtime.request_stop() for runtime in runtimes],
            )

    try:
        await asyncio.gather(*(runtime.run() for runtime in runtimes))
    finally:
        await container.shutdown()


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
