from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.modules.document.domain.models import CaseRecord
from src.modules.document.infra.persistence.orm import CaseRow


class SqlAlchemyCaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        *,
        org_id: int,
        case_id: int,
    ) -> CaseRecord | None:
        statement = select(CaseRow).where(
            CaseRow.org_id == org_id,
            CaseRow.id == case_id,
        )
        row = await self._session.scalar(statement)

        if row is None:
            return None

        return CaseRecord(
            id=row.id,
            org_id=row.org_id,
        )