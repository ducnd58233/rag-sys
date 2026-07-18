from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.document.domain.models import (
    DocumentProcessingStatus,
    DocumentScanStatus,
    DocumentVersionRecord,
)
from src.modules.document.infra.persistence.orm import DocumentVersionRow


class SqlAlchemyDocumentVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        *,
        org_id: int,
        document_version_id: int,
    ) -> DocumentVersionRecord | None:
        statement = select(DocumentVersionRow).where(
            DocumentVersionRow.id == document_version_id,
            DocumentVersionRow.org_id == org_id,
        )
        row = await self._session.scalar(statement)

        if row is None:
            return None

        return self._to_record(row)

    async def create(
        self,
        record: DocumentVersionRecord,
    ) -> None:
        self._session.add(
            DocumentVersionRow(
                id=record.id,
                org_id=record.org_id,
                document_id=record.document_id,
                storage_object_id=record.storage_object_id,
                version_no=record.version_no,
                filename=record.filename,
                mime_type=record.mime_type,
                doc_type=record.doc_type,
                doc_type_confidence=record.doc_type_confidence,
                processing_status=record.processing_status.value,
                scan_status=record.scan_status.value,
                uploaded_by=record.uploaded_by,
                page_count=None,
                created_at=datetime.now(timezone.utc),
                superseded_at=None,
            )
        )

    async def next_version_number(
        self,
        *,
        org_id: int,
        document_id: int,
    ) -> int:
        statement = select(
            func.coalesce(
                func.max(DocumentVersionRow.version_no),
                0,
            )
        ).where(
            DocumentVersionRow.org_id == org_id,
            DocumentVersionRow.document_id == document_id,
        )
        current = await self._session.scalar(statement)
        return int(current or 0) + 1

    async def set_processing_status(
        self,
        *,
        org_id: int,
        document_version_id: int,
        status: DocumentProcessingStatus,
    ) -> None:
        statement = (
            update(DocumentVersionRow)
            .where(
                DocumentVersionRow.id == document_version_id,
                DocumentVersionRow.org_id == org_id,
            )
            .values(processing_status=status.value)
        )
        await self._session.execute(statement)

    def _to_record(
        self,
        row: DocumentVersionRow,
    ) -> DocumentVersionRecord:
        return DocumentVersionRecord(
            id=row.id,
            org_id=row.org_id,
            document_id=row.document_id,
            storage_object_id=row.storage_object_id,
            version_no=row.version_no,
            filename=row.filename,
            mime_type=row.mime_type,
            processing_status=DocumentProcessingStatus(
                row.processing_status,
            ),
            scan_status=DocumentScanStatus(row.scan_status),
            uploaded_by=row.uploaded_by,
            doc_type=row.doc_type,
            doc_type_confidence=row.doc_type_confidence,
        )