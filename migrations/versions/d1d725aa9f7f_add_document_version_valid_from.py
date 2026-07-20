"""add document version valid from

Revision ID: d1d725aa9f7f
Revises: 5954d318330d
Create Date: 2026-07-20 00:00:00+00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d1d725aa9f7f"
down_revision: str | None = "5954d318330d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add version validity start for temporal retrieval."""
    statements = (
        "ALTER TABLE document_versions ADD COLUMN valid_from TIMESTAMPTZ;",
        """
        UPDATE document_versions
        SET valid_from = created_at
        WHERE valid_from IS NULL
            AND processing_status = 'indexed';
        """,
        """
        ALTER TABLE document_versions
        ADD CONSTRAINT ck_document_versions_validity CHECK (
            valid_from IS NULL
            OR superseded_at IS NULL
            OR superseded_at >= valid_from
        );
        """,
        """
        CREATE INDEX ix_document_versions_org_validity
            ON document_versions (org_id, document_id, valid_from, superseded_at);
        """,
    )

    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    """Remove version validity start."""
    statements = (
        "DROP INDEX IF EXISTS ix_document_versions_org_validity;",
        """
        ALTER TABLE document_versions
        DROP CONSTRAINT IF EXISTS ck_document_versions_validity;
        """,
        "ALTER TABLE document_versions DROP COLUMN IF EXISTS valid_from;",
    )

    for statement in statements:
        op.execute(statement)
