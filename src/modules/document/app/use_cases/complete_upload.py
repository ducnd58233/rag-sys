from __future__ import annotations

import logging
import re

from src.modules.document.app.dto import CompleteUploadRequest, CompleteUploadResult
from src.modules.document.app.ports import (
    IDocumentUnitOfWork,
    IIngestionRequestPublisher,
)
from src.modules.document.domain.errors import (
    DocumentConflictError,
    DocumentInternalError,
    DocumentNotFoundError,
    DocumentValidationError,
)
from src.modules.document.domain.models import (
    DocumentRecord,
    DocumentVersionRecord,
    StoredObjectStatus,
)
from src.shared.app.ports.message_queue import MessageQueueError
from src.shared.app.ports.object_storage import (
    IObjectStorage,
    ObjectStorageError,
    StorageBucket,
)

logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CompleteDocumentUploadUseCase:
    def __init__(
        self,
        *,
        document_uow: IDocumentUnitOfWork,
        object_storage: IObjectStorage,
        ingestion_publisher: IIngestionRequestPublisher,
    ) -> None:
        self._document_uow = document_uow
        self._object_storage = object_storage
        self._ingestion_publisher = ingestion_publisher

    async def execute(
        self,
        request: CompleteUploadRequest,
    ) -> CompleteUploadResult:
        checksum = request.checksum_sha256.strip().lower()
        if _SHA256_PATTERN.fullmatch(checksum) is None:
            raise DocumentValidationError(
                message=("checksum_sha256 must contain " "64 hexadecimal characters"),
            )

        async with self._document_uow.begin() as transaction:
            version = await transaction.document_versions.get(
                org_id=request.org_id,
                document_version_id=request.document_version_id,
            )
            if version is None:
                raise DocumentNotFoundError(
                    message="Document version was not found",
                )

            document = await transaction.documents.get(
                org_id=request.org_id,
                document_id=version.document_id,
            )
            if document is None:
                raise DocumentNotFoundError(
                    message="Document was not found",
                )

            stored_object = await transaction.stored_objects.get(
                org_id=request.org_id,
                stored_object_id=version.storage_object_id,
            )
            if stored_object is None:
                raise DocumentNotFoundError(
                    message="Stored object metadata was not found",
                )

        if stored_object.status is StoredObjectStatus.AVAILABLE:
            if stored_object.checksum_sha256 != checksum:
                raise DocumentConflictError(
                    message="Upload was completed with another checksum",
                )

            await self._request_ingestion(document, version)

            return CompleteUploadResult(
                document_id=document.id,
                document_version_id=version.id,
                version_no=version.version_no,
            )

        if stored_object.status is not StoredObjectStatus.PENDING:
            raise DocumentConflictError(
                message="Stored object cannot be completed",
            )

        try:
            bucket = StorageBucket(stored_object.bucket)
            actual_object = await self._object_storage.stat(
                bucket=bucket,
                key=stored_object.object_key,
            )
        except (ValueError, ObjectStorageError) as error:
            raise DocumentInternalError(
                message="Could not verify uploaded object",
            ) from error

        if actual_object.size_bytes != stored_object.size_bytes:
            raise DocumentConflictError(
                message=("Uploaded object size does not match " "the requested size"),
            )

        actual_content_type = (
            actual_object.content_type.split(";", maxsplit=1)[0].strip().lower()
            if actual_object.content_type is not None
            else None
        )
        if actual_content_type != stored_object.content_type.lower():
            raise DocumentConflictError(
                message=(
                    "Uploaded object content type does not match " "the requested type"
                ),
            )

        async with self._document_uow.begin() as transaction:
            was_marked_available = await transaction.stored_objects.mark_available(
                org_id=request.org_id,
                stored_object_id=stored_object.id,
                etag=actual_object.etag,
                checksum_sha256=checksum,
            )
            if not was_marked_available:
                raise DocumentConflictError(
                    message="Upload completion state changed",
                )

            await transaction.documents.set_current_version(
                org_id=request.org_id,
                document_id=document.id,
                document_version_id=version.id,
            )

        await self._request_ingestion(document, version)

        return CompleteUploadResult(
            document_id=document.id,
            document_version_id=version.id,
            version_no=version.version_no,
        )

    async def _request_ingestion(
        self,
        document: DocumentRecord,
        version: DocumentVersionRecord,
    ) -> None:
        try:
            await self._ingestion_publisher.request_ingestion(
                org_id=document.org_id,
                document_id=document.id,
                document_version_id=version.id,
                version_no=version.version_no,
            )
        except MessageQueueError:
            logger.warning(
                "Could not publish ingestion request; "
                "version stays UPLOADED for reconciliation",
                extra={"document_version_id": version.id},
            )
