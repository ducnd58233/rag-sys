"""Shared constants for the evaluation scripts.

Kept in its own module (rather than defined in index_corpus.py) so run.py, which only
needs an HTTP client, does not have to import src.bootstrap and its heavy dependency
graph (SQLAlchemy, Kafka, Neo4j, MinIO clients) just to read a default org_id.
"""

from __future__ import annotations

DEFAULT_EVAL_ORG_ID = 999_000
DEFAULT_EVAL_USER_ID = 999_001
