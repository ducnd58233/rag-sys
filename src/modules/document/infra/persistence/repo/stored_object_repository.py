from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.document.domain.models import (
    StoredObjectRecord,
    StoredObjectStatus,
)
from src.modules.document.infra.persistence.orm import StoredObjectRow


class SqlAlchemyStoredObjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        *,
        org_id: int,
        stored_object_id: int,
    ) -> StoredObjectRecord | None:
        statement = select(StoredObjectRow).where(
            StoredObjectRow.id == stored_object_id,
            StoredObjectRow.org_id == org_id,
        )
        row = await self._session.scalar(statement)

        if row is None:
            return None

        return self._to_record(row)

    async def create(
        self,
        record: StoredObjectRecord,
    ) -> None:
        self._session.add(
            StoredObjectRow(
                id=record.id,
                org_id=record.org_id,
                bucket=record.bucket,
                object_key=record.object_key,
                purpose=record.purpose,
                content_type=record.content_type,
                size_bytes=record.size_bytes,
                checksum_sha256=record.checksum_sha256,
                etag=record.etag,
                status=record.status.value,
                created_by=record.created_by,
                created_at=datetime.now(timezone.utc),
                available_at=None,
                deleted_at=None,
            )
        )

    async def mark_available(
        self,
        *,
        org_id: int,
        stored_object_id: int,
        etag: str,
    ) -> None:
        statement = (
            update(StoredObjectRow)
            .where(
                StoredObjectRow.id == stored_object_id,
                StoredObjectRow.org_id == org_id,
            )
            .values(
                status=StoredObjectStatus.AVAILABLE.value,
                etag=etag,
                available_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(statement)

    async def mark_failed(
        self,
        *,
        org_id: int,
        stored_object_id: int,
    ) -> None:
        statement = (
            update(StoredObjectRow)
            .where(
                StoredObjectRow.id == stored_object_id,
                StoredObjectRow.org_id == org_id,
            )
            .values(status=StoredObjectStatus.FAILED.value)
        )
        await self._session.execute(statement)

    def _to_record(
        self,
        row: StoredObjectRow,
    ) -> StoredObjectRecord:
        return StoredObjectRecord(
            id=row.id,
            org_id=row.org_id,
            bucket=row.bucket,
            object_key=row.object_key,
            purpose=row.purpose,
            content_type=row.content_type,
            size_bytes=row.size_bytes,
            checksum_sha256=row.checksum_sha256,
            etag=row.etag,
            status=StoredObjectStatus(row.status),
            created_by=row.created_by,
        )