from __future__ import annotations

from collections.abc import Sequence

from aiokafka import AIOKafkaConsumer as _AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.structs import TopicPartition

from src.shared.app.ports.message_queue import (
    ConsumerGroup,
    IncomingMessage,
    MessageQueueError,
    Topic,
)
from src.shared.configs.settings import KafkaSettings


class AioKafkaConsumer:
    def __init__(
        self,
        settings: KafkaSettings,
        *,
        topic: Topic,
        group: ConsumerGroup,
    ) -> None:
        self._topic = topic
        self._consumer = _AIOKafkaConsumer(
            topic.value,
            bootstrap_servers=settings.bootstrap_servers,
            group_id=group.value,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_interval_ms=settings.max_poll_interval_ms,
            session_timeout_ms=settings.session_timeout_ms,
            max_poll_records=1,
        )

    async def start(self) -> None:
        try:
            await self._consumer.start()
        except KafkaError as error:
            raise MessageQueueError(
                f"Could not start consumer for {self._topic.value}",
            ) from error

    async def poll(self, *, timeout_ms: int) -> Sequence[IncomingMessage]:
        try:
            batches = await self._consumer.getmany(
                timeout_ms=timeout_ms,
                max_records=1,
            )
        except KafkaError as error:
            raise MessageQueueError(f"Could not poll {self._topic.value}") from error

        return [
            IncomingMessage(
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                key=record.key.decode("utf-8") if record.key is not None else None,
                payload=record.value,
            )
            for records in batches.values()
            for record in records
        ]

    async def commit(self, message: IncomingMessage) -> None:
        partition = TopicPartition(message.topic, message.partition)
        try:
            await self._consumer.commit({partition: message.offset + 1})
        except KafkaError as error:
            raise MessageQueueError("Could not commit offset") from error

    async def stop(self) -> None:
        await self._consumer.stop()
