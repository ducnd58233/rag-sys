"""create_persistence_foundation

Revision ID: 6930c9de6217
Revises: 
Create Date: 2026-07-18 10:23:27.408937+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6930c9de6217'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    statements = (
        """
        CREATE TABLE IF NOT EXISTS plans (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_plans_code CHECK (
                code ~ '^[a-z][a-z0-9_-]{1,63}$'
            ),
            CONSTRAINT uq_plans_code UNIQUE (code)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            name TEXT NOT NULL,
            region TEXT NOT NULL DEFAULT 'default',
            settings JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS organization_plan_assignments (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            plan_id BIGINT NOT NULL CHECK (plan_id > 0),
            assigned_by BIGINT CHECK (
                assigned_by IS NULL OR assigned_by > 0
            ),
            reason TEXT,
            starts_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            ends_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_org_plan_assignment_period CHECK (
                ends_at IS NULL OR ends_at > starts_at
            )
        );
        """,
        """
        CREATE UNIQUE INDEX uq_org_plan_assignments_current
            ON organization_plan_assignments (org_id)
            WHERE ends_at IS NULL;
        """,
        """
        CREATE INDEX ix_org_plan_assignments_history
            ON organization_plan_assignments (
                org_id,
                starts_at DESC
            );
        """,
        """
        CREATE INDEX ix_org_plan_assignments_plan
            ON organization_plan_assignments (plan_id);
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            email TEXT NOT NULL,
            external_auth_id TEXT,
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_users_email UNIQUE (email),
            CONSTRAINT uq_users_external_auth_id
                UNIQUE (external_auth_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS memberships (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            user_id BIGINT NOT NULL CHECK (user_id > 0),
            role TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            invited_by BIGINT CHECK (
                invited_by IS NULL OR invited_by > 0
            ),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_memberships_role CHECK (
                role IN (
                    'admin',
                    'case_manager',
                    'reviewer',
                    'viewer'
                )
            ),
            CONSTRAINT ck_memberships_status CHECK (
                status IN ('pending', 'active')
            ),
            CONSTRAINT uq_memberships_org_user
                UNIQUE (org_id, user_id)
        );
        """,
        """
        CREATE INDEX ix_memberships_org_role
            ON memberships (org_id, role);
        """,
        """
        CREATE INDEX ix_memberships_user
            ON memberships (user_id);

        """,
        """
        CREATE TABLE IF NOT EXISTS workflow_templates (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT CHECK (
                org_id IS NULL OR org_id > 0
            ),
            slug TEXT NOT NULL,
            version TEXT NOT NULL,
            definition JSONB NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            created_by BIGINT CHECK (
                created_by IS NULL OR created_by > 0
            ),
            published_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_workflow_templates_slug CHECK (
                slug ~ '^[a-z][a-z0-9-]{1,127}$'
            ),
            CONSTRAINT ck_workflow_templates_status CHECK (
                status IN ('draft', 'published', 'deprecated')
            ),
            CONSTRAINT ck_workflow_templates_published_at CHECK (
                status <> 'published'
                OR published_at IS NOT NULL
            )
        );
        """,
        """
        CREATE UNIQUE INDEX uq_workflow_templates_global_version
            ON workflow_templates (slug, version)
            WHERE org_id IS NULL;
        """,
        """
        CREATE UNIQUE INDEX uq_workflow_templates_org_version
            ON workflow_templates (org_id, slug, version)
            WHERE org_id IS NOT NULL;
        """,
        """
        CREATE INDEX ix_workflow_templates_org_status
            ON workflow_templates (org_id, status);

        """,
        """
        CREATE TABLE IF NOT EXISTS cases (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            title TEXT NOT NULL,
            client_ref TEXT,
            status TEXT NOT NULL DEFAULT 'draft',
            workflow_template_id BIGINT NOT NULL CHECK (
                workflow_template_id > 0
            ),
            template_slug TEXT NOT NULL,
            template_version TEXT NOT NULL,
            created_by BIGINT NOT NULL CHECK (created_by > 0),
            assigned_to BIGINT CHECK (
                assigned_to IS NULL OR assigned_to > 0
            ),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            archived_at TIMESTAMPTZ,
            CONSTRAINT ck_cases_status CHECK (
                status IN (
                    'draft',
                    'processing',
                    'in_review',
                    'approved',
                    'exported',
                    'archived'
                )
            ),
            CONSTRAINT ck_cases_archived_at CHECK (
                status <> 'archived'
                OR archived_at IS NOT NULL
            )
        );
        """,
        """
        CREATE INDEX ix_cases_org_status
            ON cases (org_id, status);
        """,
        """
        CREATE INDEX ix_cases_org_assigned
            ON cases (org_id, assigned_to);
        """,
        """
        CREATE INDEX ix_cases_template
            ON cases (
                workflow_template_id,
                template_version
            );

        """,
        """
        CREATE TABLE IF NOT EXISTS stored_objects (
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
            CONSTRAINT ck_stored_objects_bucket CHECK (
                bucket ~ '^[a-z0-9][a-z0-9.-]{1,62}$'
            ),
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
                status <> 'available'
                OR available_at IS NOT NULL
            ),
            CONSTRAINT ck_stored_objects_deleted_at CHECK (
                status <> 'deleted'
                OR deleted_at IS NOT NULL
            )
        );
        """,
        """
        CREATE INDEX ix_stored_objects_org_purpose_status
            ON stored_objects (org_id, purpose, status);
        """,
        """
        CREATE INDEX ix_stored_objects_org_created
            ON stored_objects (org_id, created_at DESC);

        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            case_id BIGINT NOT NULL CHECK (case_id > 0),
            display_name TEXT NOT NULL,
            current_version_id BIGINT CHECK (
                current_version_id IS NULL
                OR current_version_id > 0
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
                status <> 'deleted'
                OR deleted_at IS NOT NULL
            )
        );
        """,
        """
        CREATE INDEX ix_documents_org_case_status
            ON documents (org_id, case_id, status);
        """,
        """
        CREATE INDEX ix_documents_current_version
            ON documents (current_version_id)
            WHERE current_version_id IS NOT NULL;

        """,
        """
        CREATE TABLE IF NOT EXISTS document_versions (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            document_id BIGINT NOT NULL CHECK (document_id > 0),
            storage_object_id BIGINT NOT NULL CHECK (
                storage_object_id > 0
            ),
            version_no INTEGER NOT NULL CHECK (version_no > 0),
            filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            doc_type TEXT,
            doc_type_confidence DOUBLE PRECISION,
            processing_status TEXT NOT NULL DEFAULT 'uploaded',
            scan_status TEXT NOT NULL DEFAULT 'pending',
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
                    'scanning',
                    'parsing',
                    'indexed',
                    'failed',
                    'quarantined'
                )
            ),
            CONSTRAINT ck_document_versions_scan_status CHECK (
                scan_status IN (
                    'pending',
                    'clean',
                    'infected',
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
        """
        CREATE INDEX ix_document_versions_org_doc_type
            ON document_versions (
                org_id,
                doc_type,
                processing_status
            );

        """,
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGINT PRIMARY KEY CHECK (id > 0),
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            actor_type TEXT NOT NULL,
            actor_id BIGINT CHECK (
                actor_id IS NULL OR actor_id > 0
            ),
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id BIGINT CHECK (
                entity_id IS NULL OR entity_id > 0
            ),
            input_hash TEXT,
            model_version TEXT,
            prompt_version TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_audit_logs_actor_type CHECK (
                actor_type IN ('user', 'system', 'llm')
            )
        );
        """,
        """
        CREATE INDEX ix_audit_logs_org_entity
            ON audit_logs (org_id, entity_type, entity_id);
        """,
        """
        CREATE INDEX ix_audit_logs_org_created
            ON audit_logs (org_id, created_at DESC);

        """,
        """
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            org_id BIGINT NOT NULL CHECK (org_id > 0),
            key TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            response JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_idempotency_keys
                PRIMARY KEY (org_id, key),
            CONSTRAINT ck_idempotency_keys_expiry CHECK (
                expires_at > created_at
            )
        );
        """,
        """
        CREATE INDEX ix_idempotency_keys_expires_at
            ON idempotency_keys (expires_at);
        """,
    )
    for statement in statements:
        op.execute(statement)

def downgrade() -> None:
    statements = (
        "DROP TABLE IF EXISTS idempotency_keys;",
        "DROP TABLE IF EXISTS audit_logs;",
        "DROP TABLE IF EXISTS document_versions;",
        "DROP TABLE IF EXISTS documents;",
        "DROP TABLE IF EXISTS stored_objects;",
        "DROP TABLE IF EXISTS cases;",
        "DROP TABLE IF EXISTS workflow_templates;",
        "DROP TABLE IF EXISTS memberships;",
        "DROP TABLE IF EXISTS users;",
        "DROP TABLE IF EXISTS organization_plan_assignments;",
        "DROP TABLE IF EXISTS organizations;",
        "DROP TABLE IF EXISTS plans;",
    )
    for statement in statements:
        op.execute(statement)