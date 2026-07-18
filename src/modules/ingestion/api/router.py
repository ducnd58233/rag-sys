from fastapi import APIRouter, Depends
from src.bootstrap import AppContainer
from src.modules.ingestion.api.schemas import IngestDocumentRequestBody, IngestDocumentResponse
from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.shared.http.deps import get_container
from src.shared.http.errors import ErrorResponse


router = APIRouter(prefix="/ingestions", tags=["ingestions"])

_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Document source not found"},
    422: {"model": ErrorResponse, "description": "Validation failed"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}

@router.post(
    "/documents", 
    response_model=IngestDocumentResponse, 
    responses=_ERROR_RESPONSES,
)
async def ingest_document(
    body: IngestDocumentRequestBody,
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