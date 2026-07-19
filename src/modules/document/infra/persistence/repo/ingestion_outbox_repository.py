from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.document.domain.models import (
    IngestionOutboxEventRecord,
    IngestionOutboxEventStatus,
)
from src.modules.document.infra.persistence.orm import IngestionOutboxEventRow


class SqlAlchemyIngestionOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(
        self,
        record: IngestionOutboxEventRecord,
    ) -> None:
        self._session.add(
            IngestionOutboxEventRow(
                id=record.id,
                org_id=record.org_id,
                document_id=record.document_id,
                document_version_id=record.document_version_id,
                version_no=record.version_no,
                status=record.status.value,
                created_at=datetime.now(timezone.utc),
                published_at=None,
            )
        )

    async def find_pending(
        self,
        *,
        limit: int,
    ) -> Sequence[IngestionOutboxEventRecord]:
        statement = (
            select(IngestionOutboxEventRow)
            .where(
                IngestionOutboxEventRow.status
                == IngestionOutboxEventStatus.PENDING.value,
            )
            .order_by(IngestionOutboxEventRow.created_at)
            .limit(limit)
        )
        rows = await self._session.scalars(statement)
        return [self._to_record(row) for row in rows]

    async def mark_published(
        self,
        *,
        event_id: int,
    ) -> None:
        statement = (
            update(IngestionOutboxEventRow)
            .where(
                IngestionOutboxEventRow.id == event_id,
                IngestionOutboxEventRow.status
                == IngestionOutboxEventStatus.PENDING.value,
            )
            .values(
                status=IngestionOutboxEventStatus.PUBLISHED.value,
                published_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(statement)

    def _to_record(
        self,
        row: IngestionOutboxEventRow,
    ) -> IngestionOutboxEventRecord:
        return IngestionOutboxEventRecord(
            id=row.id,
            org_id=row.org_id,
            document_id=row.document_id,
            document_version_id=row.document_version_id,
            version_no=row.version_no,
            status=IngestionOutboxEventStatus(row.status),
        )
