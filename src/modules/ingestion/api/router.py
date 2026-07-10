from fastapi import APIRouter, Depends
from src.bootstrap import AppContainer
from src.modules.ingestion.api.schemas import IngestByPathRequest, IngestDocumentResponse
from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.shared.http.deps import get_container


router = APIRouter(prefix="/ingestions", tags=["ingestions"])

@router.post("/documents/path", response_model=IngestDocumentResponse)
async def ingest_document_by_path(
    body: IngestByPathRequest,
    container: AppContainer = Depends(get_container),
) -> IngestDocumentResponse:
    result = await container.ingest_document.execute(
        IngestDocumentRequest(
            source_path=body.source_path,
            document_id=body.document_id,
            metadata=body.metadata,
        ),
    )

    return IngestDocumentResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        status=result.status,
    )