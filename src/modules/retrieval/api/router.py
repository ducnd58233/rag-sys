from __future__ import annotations

from fastapi import APIRouter, Depends

from src.bootstrap import AppContainer
from src.modules.retrieval.api.schemas import (
    RetrieveRequestBody,
    RetrieveResponse,
    RetrievedChunkResponse,
)
from src.modules.retrieval.app.dto import RetrieveRequest
from src.shared.http.deps import get_container
from src.shared.http.errors import ErrorResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    422: {"model": ErrorResponse, "description": "Validation failed"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}


@router.post(
    "/search",
    response_model=RetrieveResponse,
    responses=_ERROR_RESPONSES,
)
async def search(
    body: RetrieveRequestBody,
    container: AppContainer = Depends(get_container),
) -> RetrieveResponse:
    result = await container.retrieve.execute(
        RetrieveRequest(
            query=body.query,
            top_k=body.top_k,
            document_id=body.document_id,
        ),
    )
    return RetrieveResponse(
        query=result.query,
        items=[
            RetrievedChunkResponse(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                content=item.content,
                score=item.score,
                metadata=dict(item.metadata),
            )
            for item in result.items
        ],
    )