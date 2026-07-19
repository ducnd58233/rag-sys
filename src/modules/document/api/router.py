from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from src.bootstrap import AppContainer
from src.modules.document.api.schemas import (
    CompleteDocumentVersionRequestBody,
    CompleteDocumentVersionResponse,
    CreateDocumentRequestBody,
    DocumentUploadResponse,
)
from src.modules.document.app.dto import CompleteUploadRequest, CreateUploadUrlRequest
from src.shared.http.deps import get_container

router = APIRouter(tags=["documents"])


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    body: CreateDocumentRequestBody,
    container: AppContainer = Depends(get_container),
) -> DocumentUploadResponse:
    result = await container.create_document_upload.execute(
        CreateUploadUrlRequest(
            org_id=body.org_id,
            user_id=body.user_id,
            filename=body.filename,
            mime_type=body.mime_type,
            size_bytes=body.size_bytes,
        ),
    )

    return DocumentUploadResponse(
        document_id=result.document_id,
        document_version_id=result.document_version_id,
        version_no=result.version_no,
        signed_put_url=result.upload_url,
        expires_in_seconds=result.expires_in_seconds,
    )


@router.post(
    "/document-versions/{document_version_id}/completion",
    response_model=CompleteDocumentVersionResponse,
)
async def complete_document_version(
    body: CompleteDocumentVersionRequestBody,
    document_version_id: int = Path(gt=0),
    container: AppContainer = Depends(get_container),
) -> CompleteDocumentVersionResponse:
    result = await container.complete_document_upload.execute(
        CompleteUploadRequest(
            org_id=body.org_id,
            document_version_id=document_version_id,
            checksum_sha256=body.checksum_sha256,
        ),
    )

    return CompleteDocumentVersionResponse(
        document_id=result.document_id,
        document_version_id=result.document_version_id,
        version_no=result.version_no,
    )
