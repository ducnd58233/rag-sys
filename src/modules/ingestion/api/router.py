from fastapi import APIRouter, Depends, Path

from src.bootstrap import AppContainer
from src.modules.ingestion.api.schemas import (
    IngestDocumentResponse,
    StartIngestionRequestBody,
)
from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.shared.http.deps import get_container

router = APIRouter(tags=["ingestions"])


@router.post(
    "/document-versions/{document_version_id}/ingestions",
    response_model=IngestDocumentResponse,
)
async def ingest_document(
    body: StartIngestionRequestBody,
    document_version_id: int = Path(gt=0),
    container: AppContainer = Depends(get_container),
) -> IngestDocumentResponse:
    result = await container.ingest_document.execute(
        IngestDocumentRequest(
            org_id=body.org_id,
            document_version_id=document_version_id,
        ),
    )
    return IngestDocumentResponse(
        document_id=result.document_id,
        document_version_id=result.document_version_id,
        version_no=result.version_no,
        chunk_count=result.chunk_count,
        status=result.status.value,
    )
