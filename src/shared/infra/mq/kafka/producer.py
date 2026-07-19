from __future__ import annotations

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from src.shared.app.ports.message_queue import MessageQueueError, Topic
from src.shared.configs.settings import KafkaSettings


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
        topic: Topic,
        *,
        key: str | None,
        payload: bytes,
    ) -> None:
        try:
            await self._producer.send_and_wait(
                topic.value,
                value=payload,
                key=key.encode("utf-8") if key is not None else None,
            )
        except KafkaError as error:
            raise MessageQueueError(f"Could not publish to {topic.value}") from error

    async def close(self) -> None:
        await self._producer.stop()
