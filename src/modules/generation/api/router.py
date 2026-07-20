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

router = APIRouter(prefix="/generation", tags=["generation"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequestBody,
    container: AppContainer = Depends(get_container),
) -> AskResponse:
    result = await container.answer_question.execute(
        AskRequest(
            org_id=body.org_id,
            query=body.query,
            top_k=body.top_k,
            document_id=body.document_id,
            document_version_id=body.document_version_id,
            as_of=body.as_of,
        ),
    )
    return AskResponse(
        query=result.query,
        answer=result.answer,
        refused=result.refused,
        citations=[
            CitationResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                score=c.score,
            )
            for c in result.citations
        ],
    )
