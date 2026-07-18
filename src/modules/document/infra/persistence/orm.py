from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.infra.database.base import Base


class CaseRow(Base):
    __tablename__ = 'cases'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(BigInteger)
    title: Mapped[str] = mapped_column(Text)
    client_ref: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    workflow_template_id: Mapped[int] = mapped_column(BigInteger)
    template_slug: Mapped[str] = mapped_column(String)
    template_version: Mapped[str] = mapped_column(String)
    created_by: Mapped[int] = mapped_column(BigInteger)
    assigned_to: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )


class DocumentRow(Base):
    __tablename__ = 'documents'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(BigInteger)
    case_id: Mapped[int] = mapped_column(BigInteger)
    display_name: Mapped[str] = mapped_column(Text)
    current_version_id: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String)
    created_by: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )


class DocumentVersionRow(Base):
    __tablename__ = 'document_versions'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(BigInteger)
    document_id: Mapped[int] = mapped_column(BigInteger)
    storage_object_id: Mapped[int] = mapped_column(BigInteger)
    version_no: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str | None] = mapped_column(Text)
    doc_type_confidence: Mapped[float | None] = mapped_column(Float)
    processing_status: Mapped[str] = mapped_column(String)
    scan_status: Mapped[str] = mapped_column(String)
    uploaded_by: Mapped[int] = mapped_column(BigInteger)
    page_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )


class StoredObjectRow(Base):
    __tablename__ = 'stored_objects'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(BigInteger)
    bucket: Mapped[str] = mapped_column(Text)
    object_key: Mapped[str] = mapped_column(Text)
    purpose: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    etag: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    created_by: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )