from fastapi import APIRouter, Depends

from src.bootstrap import AppContainer
from src.modules.ingestion.api.schemas import (
    CreateIngestionRequestBody,
    IngestDocumentResponse,
)
from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.shared.http.deps import get_container

router = APIRouter(tags=["ingestions"])


@router.post(
    "/ingestions",
    response_model=IngestDocumentResponse,
)
async def create_ingestion(
    body: CreateIngestionRequestBody,
    container: AppContainer = Depends(get_container),
) -> IngestDocumentResponse:
    result = await container.ingest_document.execute(
        IngestDocumentRequest(
            org_id=body.org_id,
            document_version_id=body.document_version_id,
        ),
    )
    return IngestDocumentResponse(
        document_id=result.document_id,
        document_version_id=result.document_version_id,
        version_no=result.version_no,
        chunk_count=result.chunk_count,
        status=result.status.value,
    )
