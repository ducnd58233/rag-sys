from __future__ import annotations

import re

from src.modules.document.app.dto import CreateUploadUrlRequest, CreateUploadUrlResult
from src.modules.document.app.ports import IDocumentUnitOfWork
from src.modules.document.domain.errors import (
    DocumentInternalError,
    DocumentValidationError,
)
from src.modules.document.domain.models import (
    DocumentProcessingStatus,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionRecord,
    StoredObjectPurpose,
    StoredObjectRecord,
    StoredObjectStatus,
)
from src.shared.app.ports import IIdGenerator, IObjectStorage, ObjectStorageError
from src.shared.app.ports.object_storage import StorageBucket

_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class CreateDocumentUploadUrlUseCase:
    def __init__(
        self,
        *,
        document_uow: IDocumentUnitOfWork,
        object_storage: IObjectStorage,
        id_generator: IIdGenerator,
        allowed_mime_types: frozenset[str],
        max_upload_size_bytes: int,
        upload_url_ttl_seconds: int,
    ) -> None:
        self._document_uow = document_uow
        self._object_storage = object_storage
        self._id_generator = id_generator
        self._allowed_mime_types = allowed_mime_types
        self._max_upload_size_bytes = max_upload_size_bytes
        self._upload_url_ttl_seconds = upload_url_ttl_seconds

    async def execute(
        self,
        request: CreateUploadUrlRequest,
    ) -> CreateUploadUrlResult:
        filename = self._sanitize_filename(request.filename)
        mime_type = request.mime_type.strip().lower()

        self._validate_request(request, mime_type)

        document_id = self._id_generator.next_id()
        document_version_id = self._id_generator.next_id()
        stored_object_id = self._id_generator.next_id()
        version_no = 1

        object_key = (
            f"org/{request.org_id}/document/{document_id}/v{version_no}/"
            f"original/{filename}"
        )

        async with self._document_uow.begin() as transaction:
            await transaction.documents.create(
                DocumentRecord(
                    id=document_id,
                    org_id=request.org_id,
                    display_name=filename,
                    current_version_id=None,
                    status=DocumentStatus.ACTIVE,
                    created_by=request.user_id,
                ),
            )
            await transaction.stored_objects.create(
                StoredObjectRecord(
                    id=stored_object_id,
                    org_id=request.org_id,
                    bucket=StorageBucket.DOCUMENTS.value,
                    object_key=object_key,
                    purpose=StoredObjectPurpose.DOCUMENT_ORIGINAL.value,
                    content_type=mime_type,
                    size_bytes=request.size_bytes,
                    checksum_sha256=None,
                    etag=None,
                    status=StoredObjectStatus.PENDING,
                    created_by=request.user_id,
                ),
            )
            await transaction.flush()
            await transaction.document_versions.create(
                DocumentVersionRecord(
                    id=document_version_id,
                    org_id=request.org_id,
                    document_id=document_id,
                    storage_object_id=stored_object_id,
                    version_no=version_no,
                    filename=filename,
                    mime_type=mime_type,
                    processing_status=(DocumentProcessingStatus.UPLOADED),
                    uploaded_by=request.user_id,
                    doc_type=None,
                    doc_type_confidence=None,
                ),
            )

        try:
            upload_url = await self._object_storage.presign_put(
                bucket=StorageBucket.DOCUMENTS,
                key=object_key,
                ttl_seconds=self._upload_url_ttl_seconds,
            )
        except ObjectStorageError as error:
            await self._mark_upload_failed(
                org_id=request.org_id,
                document_version_id=document_version_id,
                stored_object_id=stored_object_id,
            )
            raise DocumentInternalError(
                message="Could not create document upload URL",
            ) from error

        return CreateUploadUrlResult(
            document_id=document_id,
            document_version_id=document_version_id,
            version_no=version_no,
            upload_url=upload_url,
            expires_in_seconds=self._upload_url_ttl_seconds,
        )

    def _validate_request(
        self,
        request: CreateUploadUrlRequest,
        mime_type: str,
    ) -> None:
        identifiers = (
            request.org_id,
            request.user_id,
        )
        if any(identifier <= 0 for identifier in identifiers):
            raise DocumentValidationError(
                message="Organization and user IDs must be positive",
            )

        if mime_type not in self._allowed_mime_types:
            raise DocumentValidationError(
                message=f"Unsupported document MIME type: {mime_type}",
            )

        if not 0 < request.size_bytes <= self._max_upload_size_bytes:
            raise DocumentValidationError(
                message="Document size is outside the allowed range",
            )

    def _sanitize_filename(self, filename: str) -> str:
        normalized = filename.strip()

        if (
            not normalized
            or "\x00" in normalized
            or "/" in normalized
            or "\\" in normalized
        ):
            raise DocumentValidationError(
                message="Document filename is invalid",
            )

        safe_filename = _SAFE_FILENAME_PATTERN.sub("_", normalized)
        if safe_filename in {".", ".."}:
            raise DocumentValidationError(
                message="Document filename is invalid",
            )

        return safe_filename

    async def _mark_upload_failed(
        self,
        *,
        org_id: int,
        document_version_id: int,
        stored_object_id: int,
    ) -> None:
        async with self._document_uow.begin() as transaction:
            await transaction.stored_objects.mark_failed(
                org_id=org_id,
                stored_object_id=stored_object_id,
            )
            await transaction.document_versions.set_processing_status(
                org_id=org_id,
                document_version_id=document_version_id,
                status=DocumentProcessingStatus.FAILED,
            )
