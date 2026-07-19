from fastapi import APIRouter, Depends, status

from src.bootstrap import AppContainer
from src.modules.ingestion.api.schemas import (
    CreateIngestionRequestBody,
    IngestionAcceptedResponse,
)
from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.shared.http.deps import get_container

router = APIRouter(tags=["ingestions"])


@router.post(
    "/ingestions",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionAcceptedResponse,
)
async def create_ingestion(
    body: CreateIngestionRequestBody,
    container: AppContainer = Depends(get_container),
) -> IngestionAcceptedResponse:
    result = await container.request_ingestion.execute(
        IngestDocumentRequest(
            org_id=body.org_id,
            document_version_id=body.document_version_id,
        ),
    )
    return IngestionAcceptedResponse(
        document_id=result.document_id,
        document_version_id=result.document_version_id,
        version_no=result.version_no,
        status=result.processing_status.value,
    )
