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
    return clauses


def _metadata_as_str_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}

    return {key: str(value) for key, value in raw.items()}
