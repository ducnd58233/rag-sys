from __future__ import annotations

from src.shared.app.contracts.ingestion import IngestionRequestedEvent
from src.shared.app.ports.message_queue import IMessagePublisher, Topic


class KafkaIngestionRequestPublisher:
    def __init__(self, publisher: IMessagePublisher) -> None:
        self._publisher = publisher

    async def request_ingestion(
        self,
        *,
        org_id: int,
        document_id: int,
        document_version_id: int,
        version_no: int,
    ) -> None:
        event = IngestionRequestedEvent(
            org_id=org_id,
            document_id=document_id,
            document_version_id=document_version_id,
            version_no=version_no,
        )
        await self._publisher.publish(
            Topic.DOCUMENT_INGESTION_REQUESTED,
            key=f"{org_id}:{document_id}",
            payload=event.encode(),
        )
