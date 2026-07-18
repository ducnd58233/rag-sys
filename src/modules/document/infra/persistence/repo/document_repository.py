from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from src.modules.document.domain.models import DocumentRecord
from src.modules.document.infra.persistence.orm import DocumentRow

class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        *,
        org_id: int,
        document_id: int,
    ) -> DocumentRecord | None:
        statement = select(DocumentRow).where(
            DocumentRow.org_id == org_id,
            DocumentRow.id == document_id,
        )
        row = await self._session.scalar(statement)
        
        if row is None:
            return None
            
        return self._to_record(row)

    async def create(self, record: DocumentRecord) -> None:
        now = datetime.now(timezone.utc)

        self._session.add(
            DocumentRow(
                id=record.id,
                org_id=record.org_id,
                case_id=record.case_id,
                display_name=record.display_name,
                current_version_id=record.current_version_id,
                status=record.status,
                created_by=record.created_by,
                created_at=now,
                updated_at=now,
                deleted_at=None,
            )
        )

    async def set_current_version(
        self,
        *,
        org_id: int,
        document_id: int,
        document_version_id: int,
    ) -> None:
        statement = update(DocumentRow).where(
            DocumentRow.org_id == org_id,
            DocumentRow.id == document_id,
        ).values(
            current_version_id=document_version_id,
            updated_at=datetime.now(timezone.utc),
        ).returning(DocumentRow)

        await self._session.execute(statement)
    
    def _to_record(self, row: DocumentRow) -> DocumentRecord:
        return DocumentRecord(
            id=row.id,
            org_id=row.org_id,
            case_id=row.case_id,
            display_name=row.display_name,
            current_version_id=row.current_version_id,
            status=row.status,
            created_by=row.created_by,
        )