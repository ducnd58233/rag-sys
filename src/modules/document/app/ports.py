from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from src.modules.document.domain.models import (
    CaseRecord,
    DocumentProcessingStatus,
    DocumentRecord,
    DocumentVersionRecord,
    StoredObjectRecord,
)

class ICaseRepository(Protocol):
    async def get(
        self,
        *,
        org_id: int,
        case_id: int,
    ) -> CaseRecord | None: ...


class IDocumentRepository(Protocol):
    async def get(
        self,
        *,
        org_id: int,
        document_id: int,
    ) -> DocumentRecord | None: ...

    async def create(self, record: DocumentRecord) -> None: ...

    async def set_current_version(
        self,
        *,
        org_id: int,
        document_id: int,
        document_version_id: int,
    ) -> None: ...


class IDocumentVersionRepository(Protocol):
    async def get(
        self,
        *,
        org_id: int,
        document_version_id: int,
    ) -> DocumentVersionRecord | None: ...

    async def create(
        self,
        record: DocumentVersionRecord,
    ) -> None: ...

    async def next_version_number(
        self,
        *,
        org_id: int,
        document_id: int,
    ) -> int: ...

    async def set_processing_status(
        self,
        *,
        org_id: int,
        document_version_id: int,
        status: DocumentProcessingStatus,
    ) -> None: ...


class IStoredObjectRepository(Protocol):
    async def get(
        self,
        *,
        org_id: int,
        stored_object_id: int,
    ) -> StoredObjectRecord | None: ...

    async def create(
        self,
        record: StoredObjectRecord,
    ) -> None: ...

    async def mark_available(
        self,
        *,
        org_id: int,
        stored_object_id: int,
        etag: str,
    ) -> None: ...

    async def mark_failed(
        self,
        *,
        org_id: int,
        stored_object_id: int,
    ) -> None: ...


class IDocumentTransaction(Protocol):
    cases: ICaseRepository
    documents: IDocumentRepository
    document_versions: IDocumentVersionRepository
    stored_objects: IStoredObjectRepository


class IDocumentUnitOfWork(Protocol):
    def begin(
        self,
    ) -> AbstractAsyncContextManager[IDocumentTransaction]: ...