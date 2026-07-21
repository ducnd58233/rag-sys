from __future__ import annotations

RUBRIC_VERSION = "judge-rubrics-v1"

CLAIM_DECOMPOSITION_SYSTEM = (
    "Decompose the answer into atomic factual claims. Each claim must be a single, "
    "self-contained, checkable assertion. Split compound sentences. Do not add, "
    "infer, or omit information beyond what the answer states. Return an empty "
    "list if the answer contains no checkable claims."
)


def claim_decomposition_user(answer: str) -> str:
    return f"Answer:\n{answer}"


FAITHFULNESS_SYSTEM = (
    "You are grading faithfulness: whether each claim is supported by the given "
    "context, using only the context, not outside knowledge. For each claim index "
    "not supported by the context, include it in unsupported_claim_indices. A "
    "claim is unsupported if the context does not state it or contradicts it."
)


def faithfulness_user(*, claims: list[str], context: str) -> str:
    numbered_claims = "\n".join(f"{i}: {claim}" for i, claim in enumerate(claims))
    return f"Context:\n{context}\n\nClaims:\n{numbered_claims}"


RELEVANCY_SYSTEM = (
    "Score 0-4 how directly the answer addresses the question, regardless of "
    "whether the answer is factually correct.\n"
    "4: Directly and completely addresses the question.\n"
    "3: Addresses the question with minor omissions or tangents.\n"
    "2: Partially addresses the question.\n"
    "1: Only loosely related to the question.\n"
    "0: Does not address the question at all."
)


def relevancy_user(*, question: str, answer: str) -> str:
    return f"Question:\n{question}\n\nAnswer:\n{answer}"


CLAIM_CORRECTNESS_SYSTEM = (
    "You are grading answer correctness against a reference. Label each generated "
    "claim 'correct' if it is consistent with (supported by or a valid "
    "paraphrase/subset of) the reference claims, or 'incorrect' if it contradicts "
    "them or is unsupported by them. Separately, list which reference claim "
    "indices are covered (semantically restated or entailed) by any generated "
    "claim. This is a semantic judgment, not a string-match."
)


def claim_correctness_user(
    *,
    generated_claims: list[str],
    reference_claims: list[str],
) -> str:
    numbered_generated = "\n".join(
        f"{i}: {claim}" for i, claim in enumerate(generated_claims)
    )
    numbered_reference = "\n".join(
        f"{i}: {claim}" for i, claim in enumerate(reference_claims)
    )
    return (
        f"Reference claims:\n{numbered_reference}\n\n"
        f"Generated claims:\n{numbered_generated}"
    )


CITATION_SUPPORT_SYSTEM = (
    "For each (claim, cited passage) pair, judge whether the passage genuinely "
    "supports the claim - not merely whether it is topically related. Include a "
    "pair's index in supported_pair_indices only when the passage's content "
    "actually substantiates the claim."
)


def citation_support_user(pairs: list[tuple[str, str]]) -> str:
    numbered_pairs = "\n\n".join(
        f"{i}: claim: {claim}\n   passage: {passage}"
        for i, (claim, passage) in enumerate(pairs)
    )
    return f"Pairs:\n{numbered_pairs}"
