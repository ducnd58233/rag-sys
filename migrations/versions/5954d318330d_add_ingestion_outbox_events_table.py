"""add ingestion outbox events table

Revision ID: 5954d318330d
Revises: 6930c9de6217
Create Date: 2026-07-19 15:54:32.515792+00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "5954d318330d"
down_revision: str | None = "6930c9de6217"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the transactional outbox table for ingestion-requested events."""
    statements = (
        """
        CREATE TABLE ingestion_outbox_events (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            document_id BIGINT NOT NULL
                REFERENCES documents (id) ON DELETE CASCADE
                CHECK (document_id > 0),
            document_version_id BIGINT NOT NULL
                REFERENCES document_versions (id) ON DELETE CASCADE
                CHECK (document_version_id > 0),
            version_no INTEGER NOT NULL CHECK (version_no > 0),
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            published_at TIMESTAMPTZ,
            CONSTRAINT ck_ingestion_outbox_events_status CHECK (
                status IN ('pending', 'published')
            ),
            CONSTRAINT ck_ingestion_outbox_events_published_at CHECK (
                status <> 'published' OR published_at IS NOT NULL
            )
        );
        """,
        """
        CREATE INDEX ix_ingestion_outbox_events_pending
            ON ingestion_outbox_events (created_at)
            WHERE status = 'pending';
        """,
    )

    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    """Drop the ingestion outbox table."""
    op.execute("DROP TABLE IF EXISTS ingestion_outbox_events;")
