from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from src.modules.document.domain.models import (
    DocumentProcessingStatus,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionRecord,
    IngestionOutboxEventRecord,
    IngestionOutboxEventStatus,
    StoredObjectRecord,
    StoredObjectStatus,
)
from src.modules.document.infra.unit_of_work import SqlAlchemyDocumentUnitOfWork
from src.shared.infra.database import Database

pytestmark = pytest.mark.integration


@pytest.fixture
def uow(database: Database) -> SqlAlchemyDocumentUnitOfWork:
    return SqlAlchemyDocumentUnitOfWork(database.session_factory)


@pytest.mark.asyncio
async def test_document_and_version_round_trip_through_real_postgres(
    uow: SqlAlchemyDocumentUnitOfWork,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1
    storage_object_id = unique_id + 2

    async with uow.begin() as tx:
        await tx.stored_objects.create(
            _stored_object(id=storage_object_id, org_id=org_id)
        )
        await tx.documents.create(_document(id=document_id, org_id=org_id))
        # See the flush note above: required before document_versions.create.
        await tx.flush()
        next_no = await tx.document_versions.next_version_number(
            org_id=org_id, document_id=document_id
        )
        assert next_no == 1
        await tx.document_versions.create(
            _document_version(
                id=unique_id + 3,
                org_id=org_id,
                document_id=document_id,
                storage_object_id=storage_object_id,
                version_no=next_no,
            )
        )

    async with uow.begin() as tx:
        document = await tx.documents.get(org_id=org_id, document_id=document_id)
        version = await tx.document_versions.get(
            org_id=org_id, document_version_id=unique_id + 3
        )
        next_no_after = await tx.document_versions.next_version_number(
            org_id=org_id, document_id=document_id
        )

    assert document is not None
    assert document.display_name == "paper.pdf"
    assert version is not None
    assert version.processing_status is DocumentProcessingStatus.UPLOADED
    assert next_no_after == 2


@pytest.mark.asyncio
async def test_document_version_lifecycle_updates_persist(
    uow: SqlAlchemyDocumentUnitOfWork,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1
    storage_object_id = unique_id + 2
    version_id = unique_id + 3
    valid_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
    superseded_at = datetime(2026, 2, 1, tzinfo=timezone.utc)

    async with uow.begin() as tx:
        await tx.stored_objects.create(
            _stored_object(id=storage_object_id, org_id=org_id)
        )
        await tx.documents.create(_document(id=document_id, org_id=org_id))
        # SQLAlchemy does not auto-order inserts across mapped classes
        # without a declared relationship(); the real production code
        # (create_upload_url.py) flushes explicitly here for the same
        # reason. Omitting this is exactly the bug a mocked session
        # would never catch, since it doesn't enforce FK ordering.
        await tx.flush()
        await tx.document_versions.create(
            _document_version(
                id=version_id,
                org_id=org_id,
                document_id=document_id,
                storage_object_id=storage_object_id,
                version_no=1,
            )
        )

    async with uow.begin() as tx:
        await tx.document_versions.set_processing_status(
            org_id=org_id,
            document_version_id=version_id,
            status=DocumentProcessingStatus.PARSING,
        )
        await tx.document_versions.activate(
            org_id=org_id,
            document_version_id=version_id,
            valid_from=valid_from,
        )

    async with uow.begin() as tx:
        version = await tx.document_versions.get(
            org_id=org_id, document_version_id=version_id
        )
    assert version is not None
    assert version.processing_status is DocumentProcessingStatus.PARSING
    assert version.valid_from == valid_from

    async with uow.begin() as tx:
        await tx.document_versions.supersede(
            org_id=org_id,
            document_version_id=version_id,
            superseded_at=superseded_at,
        )

    async with uow.begin() as tx:
        version = await tx.document_versions.get(
            org_id=org_id, document_version_id=version_id
        )
    assert version is not None
    assert version.superseded_at == superseded_at


@pytest.mark.asyncio
async def test_mark_available_only_transitions_from_pending_once(
    uow: SqlAlchemyDocumentUnitOfWork,
    unique_id: int,
) -> None:
    org_id = unique_id
    storage_object_id = unique_id + 1

    async with uow.begin() as tx:
        await tx.stored_objects.create(
            _stored_object(id=storage_object_id, org_id=org_id)
        )

    async with uow.begin() as tx:
        first = await tx.stored_objects.mark_available(
            org_id=org_id,
            stored_object_id=storage_object_id,
            etag="etag-1",
            checksum_sha256="a" * 64,
        )
    assert first is True

    # Real conditional-update semantics: the WHERE clause guards on
    # status == PENDING, so a second call against an already-AVAILABLE row
    # must be a no-op. A fake session with no real WHERE evaluation would
    # never catch a regression here.
    async with uow.begin() as tx:
        second = await tx.stored_objects.mark_available(
            org_id=org_id,
            stored_object_id=storage_object_id,
            etag="etag-2",
            checksum_sha256="b" * 64,
        )
    assert second is False

    async with uow.begin() as tx:
        record = await tx.stored_objects.get(
            org_id=org_id, stored_object_id=storage_object_id
        )
    assert record is not None
    assert record.status is StoredObjectStatus.AVAILABLE
    assert record.etag == "etag-1"


@pytest.mark.asyncio
async def test_outbox_find_pending_excludes_published_events(
    uow: SqlAlchemyDocumentUnitOfWork,
    unique_id: int,
) -> None:
    org_id = unique_id
    document_id = unique_id + 1
    storage_object_id = unique_id + 2
    version_id = unique_id + 3
    event_a_id = unique_id + 4
    event_b_id = unique_id + 5

    async with uow.begin() as tx:
        await tx.stored_objects.create(
            _stored_object(id=storage_object_id, org_id=org_id)
        )
        await tx.documents.create(_document(id=document_id, org_id=org_id))
        # SQLAlchemy does not auto-order inserts across mapped classes
        # without a declared relationship(); the real production code
        # (create_upload_url.py) flushes explicitly here for the same
        # reason. Omitting this is exactly the bug a mocked session
        # would never catch, since it doesn't enforce FK ordering.
        await tx.flush()
        await tx.document_versions.create(
            _document_version(
                id=version_id,
                org_id=org_id,
                document_id=document_id,
                storage_object_id=storage_object_id,
                version_no=1,
            )
        )
        await tx.ingestion_outbox.enqueue(
            _outbox_event(
                id=event_a_id,
                org_id=org_id,
                document_id=document_id,
                document_version_id=version_id,
            )
        )
        await tx.ingestion_outbox.enqueue(
            _outbox_event(
                id=event_b_id,
                org_id=org_id,
                document_id=document_id,
                document_version_id=version_id,
            )
        )

    async with uow.begin() as tx:
        await tx.ingestion_outbox.mark_published(event_id=event_a_id)

    async with uow.begin() as tx:
        pending = await tx.ingestion_outbox.find_pending(limit=50)

    pending_ids = {event.id for event in pending}
    assert event_a_id not in pending_ids
    assert event_b_id in pending_ids


@pytest.mark.asyncio
async def test_document_version_foreign_key_is_enforced_by_real_postgres(
    uow: SqlAlchemyDocumentUnitOfWork,
    unique_id: int,
) -> None:
    """A mocked session would happily accept this insert; a real database
    with the schema from our own migrations must reject it. This is
    precisely the class of bug unit tests with fakes cannot catch. Goes
    through the same repository/UnitOfWork path as every other test here,
    not a raw session, so it exercises the same code as production."""
    with pytest.raises(IntegrityError):
        async with uow.begin() as tx:
            await tx.document_versions.create(
                _document_version(
                    id=unique_id,
                    org_id=unique_id,
                    document_id=unique_id + 999_999,  # does not exist
                    storage_object_id=unique_id + 999_998,  # does not exist
                    version_no=1,
                )
            )


def _stored_object(*, id: int, org_id: int) -> StoredObjectRecord:
    return StoredObjectRecord(
        id=id,
        org_id=org_id,
        bucket="documents",
        object_key=f"{org_id}/{id}/paper.pdf",
        purpose="document_original",
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256=None,
        etag=None,
        status=StoredObjectStatus.PENDING,
        created_by=1,
    )


def _document(*, id: int, org_id: int) -> DocumentRecord:
    return DocumentRecord(
        id=id,
        org_id=org_id,
        display_name="paper.pdf",
        current_version_id=None,
        status=DocumentStatus.ACTIVE,
        created_by=1,
    )


def _document_version(
    *,
    id: int,
    org_id: int,
    document_id: int,
    storage_object_id: int,
    version_no: int,
) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        id=id,
        org_id=org_id,
        document_id=document_id,
        storage_object_id=storage_object_id,
        version_no=version_no,
        filename="paper.pdf",
        mime_type="application/pdf",
        processing_status=DocumentProcessingStatus.UPLOADED,
        uploaded_by=1,
        doc_type=None,
        doc_type_confidence=None,
        created_at=datetime.now(timezone.utc),
        valid_from=None,
        superseded_at=None,
    )


def _outbox_event(
    *,
    id: int,
    org_id: int,
    document_id: int,
    document_version_id: int,
) -> IngestionOutboxEventRecord:
    return IngestionOutboxEventRecord(
        id=id,
        org_id=org_id,
        document_id=document_id,
        document_version_id=document_version_id,
        version_no=1,
        status=IngestionOutboxEventStatus.PENDING,
    )
