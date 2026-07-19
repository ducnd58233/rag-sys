from __future__ import annotations

import asyncio
import signal
import sys

from src.bootstrap import build_container
from src.shared.app.mq.runtime import ConsumerRuntime
from src.shared.app.ports.message_queue import ConsumerGroup, Topic
from src.shared.infra.mq import AioKafkaConsumer


async def _run() -> None:
    container = build_container(service_name="worker")
    await container.startup()

    consumer = AioKafkaConsumer(
        container.settings.kafka,
        topic=Topic.DOCUMENT_INGESTION_REQUESTED,
        group=ConsumerGroup.INGESTION_WORKER,
    )
    runtime = ConsumerRuntime(
        consumer=consumer,
        handler=container.ingestion_handler.handle,
        poll_timeout_ms=container.settings.kafka.consumer_poll_timeout_ms,
    )

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, runtime.request_stop)

    try:
        await runtime.run()
    finally:
        await container.shutdown()


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
