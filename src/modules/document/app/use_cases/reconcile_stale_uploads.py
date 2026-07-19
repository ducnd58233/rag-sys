from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from src.modules.document.app.ports import (
    IDocumentUnitOfWork,
    IIngestionRequestPublisher,
)
from src.shared.app.ports.message_queue import MessageQueueError

logger = logging.getLogger(__name__)

_BATCH_LIMIT = 100


class ReconcileStaleUploadsUseCase:
    """Safety net for ADR-IW-004 / SPEC §5.3.

    Republishes the ingestion-requested event for document versions whose upload
    completed (``stored_object.status = AVAILABLE``) but that never advanced past
    ``processing_status = UPLOADED`` — the symptom of a Kafka publish that was lost
    after the upload transaction committed (D4).
    """

    def __init__(
        self,
        *,
        document_uow: IDocumentUnitOfWork,
        ingestion_publisher: IIngestionRequestPublisher,
        grace_period_seconds: float,
    ) -> None:
        self._document_uow = document_uow
        self._ingestion_publisher = ingestion_publisher
        self._grace_period_seconds = grace_period_seconds

    async def execute(self) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=self._grace_period_seconds)

        async with self._document_uow.begin() as transaction:
            stale_versions = await transaction.document_versions.find_stale_uploaded(
                older_than=cutoff,
                limit=_BATCH_LIMIT,
            )

        for stale in stale_versions:
            # A non-zero count here means ingestion-requested publishes are being lost.
            logger.warning(
                "Reconciliation republishing stale ingestion request",
                extra={
                    "document_version_id": stale.document_version_id,
                    "document_id": stale.document_id,
                },
            )
            try:
                await self._ingestion_publisher.request_ingestion(
                    org_id=stale.org_id,
                    document_id=stale.document_id,
                    document_version_id=stale.document_version_id,
                    version_no=stale.version_no,
                )
            except MessageQueueError:
                logger.warning(
                    "Reconciliation could not republish; will retry next sweep",
                    extra={"document_version_id": stale.document_version_id},
                )

        return len(stale_versions)
