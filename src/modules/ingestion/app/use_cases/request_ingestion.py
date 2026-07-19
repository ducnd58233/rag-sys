from __future__ import annotations

from src.modules.document.app.ports import IDocumentUnitOfWork
from src.modules.document.domain.models import (
    IngestionOutboxEventRecord,
    IngestionOutboxEventStatus,
)
from src.modules.ingestion.app.dto import IngestDocumentRequest, RequestIngestionResult
from src.modules.ingestion.domain.errors import IngestionNotFoundError
from src.shared.app.ports import IIdGenerator


class RequestIngestionUseCase:
    """Enqueues an ingestion request without running the pipeline inline (D1/FR-PROD-6).

    Kept separate from ``IngestDocumentUseCase`` (the worker-side pipeline) so the API
    process only ever writes to the outbox; it never talks to Kafka or executes the
    pipeline itself.
    """

    def __init__(
        self,
        *,
        doc_uow: IDocumentUnitOfWork,
        id_generator: IIdGenerator,
    ) -> None:
        self._doc_uow = doc_uow
        self._id_generator = id_generator

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

            await transaction.ingestion_outbox.enqueue(
                IngestionOutboxEventRecord(
                    id=self._id_generator.next_id(),
                    org_id=request.org_id,
                    document_id=document.id,
                    document_version_id=version.id,
                    version_no=version.version_no,
                    status=IngestionOutboxEventStatus.PENDING,
                ),
            )

        return RequestIngestionResult(
            document_id=document.id,
            document_version_id=version.id,
            version_no=version.version_no,
            processing_status=version.processing_status,
        )
