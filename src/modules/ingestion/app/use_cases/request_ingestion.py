from __future__ import annotations

import logging

from src.modules.document.app.ports import (
    IDocumentUnitOfWork,
    IIngestionRequestPublisher,
)
from src.modules.ingestion.app.dto import IngestDocumentRequest, RequestIngestionResult
from src.modules.ingestion.domain.errors import IngestionNotFoundError
from src.shared.app.ports.message_queue import MessageQueueError

logger = logging.getLogger(__name__)


class RequestIngestionUseCase:
    """Enqueues an ingestion request without running the pipeline inline (D1/FR-PROD-6).

    Kept separate from ``IngestDocumentUseCase`` (the worker-side pipeline) so the API
    process only ever publishes; it never executes the pipeline itself.
    """

    def __init__(
        self,
        *,
        doc_uow: IDocumentUnitOfWork,
        ingestion_publisher: IIngestionRequestPublisher,
    ) -> None:
        self._doc_uow = doc_uow
        self._ingestion_publisher = ingestion_publisher

    async def execute(
        self,
        request: IngestDocumentRequest,
    ) -> RequestIngestionResult:
        async with self._doc_uow.begin() as transaction:
            version = await transaction.document_versions.get(
                org_id=request.org_id,
                document_version_id=request.document_version_id,
            )
            if version is None:
                raise IngestionNotFoundError(
                    message="Document version not found",
                )

            document = await transaction.documents.get(
                org_id=request.org_id,
                document_id=version.document_id,
            )
            if document is None:
                raise IngestionNotFoundError(
                    message="Document source is incomplete",
                )

        try:
            await self._ingestion_publisher.request_ingestion(
                org_id=request.org_id,
                document_id=document.id,
                document_version_id=version.id,
                version_no=version.version_no,
            )
        except MessageQueueError:
            # Mirrors CompleteDocumentUploadUseCase (D4): publish failures never fail
            # the request; the reconciliation sweeper recovers a version stuck UPLOADED.
            logger.warning(
                "Could not publish ingestion request",
                extra={"document_version_id": version.id},
            )

        return RequestIngestionResult(
            document_id=document.id,
            document_version_id=version.id,
            version_no=version.version_no,
            processing_status=version.processing_status,
        )
