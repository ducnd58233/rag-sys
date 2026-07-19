from __future__ import annotations

import logging

from src.modules.document.app.ports import (
    IDocumentUnitOfWork,
    IIngestionRequestPublisher,
)
from src.shared.app.ports.message_queue import MessageQueueError

logger = logging.getLogger(__name__)

_BATCH_LIMIT = 100


class PublishIngestionOutboxEventsUseCase:
    """The relay half of the transactional outbox pattern.

    ``CompleteDocumentUploadUseCase`` and ``RequestIngestionUseCase`` only ever write
    a pending row to ``ingestion_outbox_events``, in the same local transaction as the
    business write. This use case is the sole caller of ``IIngestionRequestPublisher``:
    it polls pending rows and publishes them, marking each published only after the
    broker acknowledges it. A row left pending after a failed publish is retried on the
    next tick, so ingestion is never silently lost to a dual-write gap.
    """

    def __init__(
        self,
        *,
        document_uow: IDocumentUnitOfWork,
        ingestion_publisher: IIngestionRequestPublisher,
    ) -> None:
        self._document_uow = document_uow
        self._ingestion_publisher = ingestion_publisher

    async def execute(self) -> int:
        async with self._document_uow.begin() as transaction:
            pending_events = await transaction.ingestion_outbox.find_pending(
                limit=_BATCH_LIMIT,
            )

        for event in pending_events:
            try:
                await self._ingestion_publisher.request_ingestion(
                    org_id=event.org_id,
                    document_id=event.document_id,
                    document_version_id=event.document_version_id,
                    version_no=event.version_no,
                )
            except MessageQueueError:
                logger.warning(
                    "Could not publish outbox event; will retry next relay tick",
                    extra={"document_version_id": event.document_version_id},
                )
                continue

            async with self._document_uow.begin() as transaction:
                await transaction.ingestion_outbox.mark_published(event_id=event.id)

        return len(pending_events)
