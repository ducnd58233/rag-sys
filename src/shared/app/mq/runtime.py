from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.shared.app.ports.message_queue import IMessageConsumer, IncomingMessage

logger = logging.getLogger(__name__)

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]


class ConsumerRuntime:
    def __init__(
        self,
        *,
        consumer: IMessageConsumer,
        handler: MessageHandler,
        poll_timeout_ms: int = 1_000,
    ) -> None:
        self._consumer = consumer
        self._handler = handler
        self._poll_timeout_ms = poll_timeout_ms
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        await self._consumer.start()
        try:
            while not self._stopping.is_set():
                messages = await self._consumer.poll(
                    timeout_ms=self._poll_timeout_ms,
                )
                for message in messages:
                    await self._process(message)
        finally:
            await self._consumer.stop()

    def request_stop(self) -> None:
        self._stopping.set()

    async def _process(self, message: IncomingMessage) -> None:
        try:
            await self._handler(message)
        except Exception:
            logger.exception(
                "Message handling failed; offset will not be committed",
                extra={
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                },
            )
            return

        await self._consumer.commit(message)
