"""Create standalone document persistence tables.

Revision ID: 6930c9de6217
Revises:
Create Date: 2026-07-18 10:23:27.408937+00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "6930c9de6217"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables required by upload and ingestion."""
    statements = (
        """
        CREATE TABLE stored_objects (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            bucket TEXT NOT NULL,
            object_key TEXT NOT NULL,
            purpose TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
            checksum_sha256 CHAR(64),
            etag TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_by BIGINT CHECK (
                created_by IS NULL OR created_by > 0
            ),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            available_at TIMESTAMPTZ,
            deleted_at TIMESTAMPTZ,
            CONSTRAINT uq_stored_objects_location
                UNIQUE (bucket, object_key),
            CONSTRAINT ck_stored_objects_purpose CHECK (
                purpose IN (
                    'document_original',
                    'derived_asset',
                    'export',
                    'temporary'
                )
            ),
            CONSTRAINT ck_stored_objects_status CHECK (
                status IN (
                    'pending',
                    'available',
                    'quarantined',
                    'deleted',
                    'failed'
                )
            ),
            CONSTRAINT ck_stored_objects_checksum CHECK (
                checksum_sha256 IS NULL
                OR checksum_sha256 ~ '^[0-9a-fA-F]{64}$'
            ),
            CONSTRAINT ck_stored_objects_available_at CHECK (
                status <> 'available' OR available_at IS NOT NULL
            ),
            CONSTRAINT ck_stored_objects_deleted_at CHECK (
                status <> 'deleted' OR deleted_at IS NOT NULL
            )
        );
        """,
        """
        CREATE INDEX ix_stored_objects_org_status
            ON stored_objects (org_id, status);
        """,
        """
        CREATE TABLE documents (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            display_name TEXT NOT NULL,
            current_version_id BIGINT CHECK (
                current_version_id IS NULL OR current_version_id > 0
            ),
            status TEXT NOT NULL DEFAULT 'active',
            created_by BIGINT NOT NULL CHECK (created_by > 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            CONSTRAINT ck_documents_status CHECK (
                status IN ('active', 'deleted')
            ),
            CONSTRAINT ck_documents_deleted_at CHECK (
                status <> 'deleted' OR deleted_at IS NOT NULL
            )
        );
        """,
        """
        CREATE INDEX ix_documents_org_status
            ON documents (org_id, status);
        """,
        """
        CREATE TABLE document_versions (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            document_id BIGINT NOT NULL
                REFERENCES documents (id) ON DELETE CASCADE
                CHECK (document_id > 0),
            storage_object_id BIGINT NOT NULL CHECK (
                storage_object_id > 0
            ) REFERENCES stored_objects (id),
            version_no INTEGER NOT NULL CHECK (version_no > 0),
            filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            doc_type TEXT,
            doc_type_confidence DOUBLE PRECISION,
            processing_status TEXT NOT NULL DEFAULT 'uploaded',
            uploaded_by BIGINT NOT NULL CHECK (uploaded_by > 0),
            page_count INTEGER CHECK (
                page_count IS NULL OR page_count >= 0
            ),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            superseded_at TIMESTAMPTZ,
            CONSTRAINT uq_document_versions_number
                UNIQUE (document_id, version_no),
            CONSTRAINT uq_document_versions_storage_object
                UNIQUE (storage_object_id),
            CONSTRAINT ck_document_versions_confidence CHECK (
                doc_type_confidence IS NULL
                OR doc_type_confidence BETWEEN 0 AND 1
            ),
            CONSTRAINT ck_document_versions_processing_status CHECK (
                processing_status IN (
                    'uploaded',
                    'parsing',
                    'indexed',
                    'failed'
                )
            )
        );
        """,
        """
        CREATE INDEX ix_document_versions_org_document
            ON document_versions (
                org_id,
                document_id,
                version_no DESC
            );
        """,
    )

    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    """Drop standalone document persistence tables."""
    for statement in (
        "DROP TABLE IF EXISTS document_versions;",
        "DROP TABLE IF EXISTS documents;",
        "DROP TABLE IF EXISTS stored_objects;",
    ):
        op.execute(statement)
