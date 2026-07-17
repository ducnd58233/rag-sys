from __future__ import annotations

from fastapi import APIRouter, Depends

from src.bootstrap import AppContainer
from src.modules.generation.api.schemas import (
    AskRequestBody,
    AskResponse,
    CitationResponse,
)
from src.modules.generation.app.dto import AskRequest
from src.shared.http.deps import get_container
from src.shared.http.errors import ErrorResponse

router = APIRouter(prefix="/generation", tags=["generation"])

_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    422: {"model": ErrorResponse, "description": "Validation failed"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}


@router.post("/ask", response_model=AskResponse, responses=_ERROR_RESPONSES)
async def ask(
    body: AskRequestBody,
    container: AppContainer = Depends(get_container),
) -> AskResponse:
    result = await container.answer_question.execute(
        AskRequest(
            query=body.query,
            top_k=body.top_k,
            document_id=body.document_id,
        ),
    )
    return AskResponse(
        query=result.query,
        answer=result.answer,
        refused=result.refused,
        citations=[
            CitationResponse(chunk_id=c.chunk_id, document_id=c.document_id)
            for c in result.citations
        ],
    )