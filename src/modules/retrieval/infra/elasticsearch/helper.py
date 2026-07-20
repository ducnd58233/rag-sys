from datetime import datetime, timezone

from src.modules.retrieval.app.dto import RetrievalFilter


def build_filter_clauses(
    filters: RetrievalFilter,
) -> list[dict[str, object]]:
    clauses: list[dict[str, object]] = [
        {"term": {"org_id": str(filters.org_id)}},
    ]
    if filters.document_id is not None:
        clauses.append(
            {"term": {"document_id": str(filters.document_id)}},
        )
    if filters.document_version_id is not None:
        clauses.append(
            {
                "term": {
                    "document_version_id": str(
                        filters.document_version_id,
                    ),
                },
            },
        )
    else:
        clauses.extend(_validity_clauses(filters.as_of))
    return clauses


def _validity_clauses(as_of: datetime | None) -> list[dict[str, object]]:
    effective_at = as_of or datetime.now(timezone.utc)
    value = effective_at.astimezone(timezone.utc).isoformat()
    return [
        {"range": {"valid_from": {"lte": value}}},
        {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "must_not": [
                                {"exists": {"field": "valid_to"}},
                            ],
                        }
                    },
                    {"range": {"valid_to": {"gt": value}}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]


def _metadata_as_str_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}

    return {key: str(value) for key, value in raw.items()}
