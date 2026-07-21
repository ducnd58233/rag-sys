from __future__ import annotations


def citation_precision(*, cited_count: int, supported_count: int) -> float | None:
    """Fraction of citations whose cited passage genuinely supports its claim."""
    if cited_count == 0:
        return None
    return supported_count / cited_count


def citation_recall(
    *,
    claims_requiring_evidence_count: int,
    claims_with_valid_citation_count: int,
) -> float | None:
    """Fraction of evidence-requiring claims that carry a valid citation."""
    if claims_requiring_evidence_count == 0:
        return None
    return claims_with_valid_citation_count / claims_requiring_evidence_count
