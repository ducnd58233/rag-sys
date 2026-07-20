from __future__ import annotations

from fastapi import APIRouter, Depends

from src.bootstrap import AppContainer
from src.modules.retrieval.api.schemas import (
    RetrievalPlanResponse,
    RetrievedChunkResponse,
    RetrieveRequestBody,
    RetrieveResponse,
    StrategySelectionResponse,
)
from src.modules.retrieval.app.dto import RetrievalFilter, RetrieveRequest
from src.shared.http.deps import get_container

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrieveResponse,
)
async def search(
    body: RetrieveRequestBody,
    container: AppContainer = Depends(get_container),
) -> RetrieveResponse:
    result = await container.retrieve.execute(
        RetrieveRequest(
            query=body.query,
            top_k=body.top_k,
            filters=RetrievalFilter(
                org_id=body.org_id,
                document_id=body.document_id,
                document_version_id=body.document_version_id,
                as_of=body.as_of,
            ),
        ),
    )
    return RetrieveResponse(
        query=result.query,
        plan=RetrievalPlanResponse(
            strategies=[
                StrategySelectionResponse(
                    name=strategy.name,
                    weight=strategy.weight,
                    top_k=strategy.top_k,
                )
                for strategy in result.plan.strategies
            ],
            router_kind=result.plan.router_kind.value,
            reason=result.plan.reason,
            confidence=result.plan.confidence,
        ),
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
