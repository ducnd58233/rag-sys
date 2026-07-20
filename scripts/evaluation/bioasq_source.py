from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from scripts.evaluation.dataset import EvalCase, Split, case_from_dict

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TEST_SPLIT_FRACTION = 0.2
_SOURCE_URL = "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq"
_SOURCE_LICENSE = "cc-by-2.5"

# BioASQ ships binary relevance judgments only, not the 0-3 graded scale ndcg_at_k was
# designed around, so every relevant id gets the same grade and nDCG here is a
# binary-relevance variant, not a true multi-grade signal.
_UNIFORM_RELEVANCE_GRADE = 1


def _split_claims(answer: str) -> list[str]:
    sentences = [
        s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()
    ]
    return sentences or [answer.strip()]


def _split_for(case_id: str, fraction: float) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < fraction else "dev"


def _row_to_case_dict(row: dict[str, object]) -> dict[str, object]:
    case_id = f"bioasq-qa-{row['id']}"
    relevant_passage_ids: list[int] = json.loads(row["relevant_passage_ids"])  # type: ignore[arg-type]
    relevant_document_ids = [f"bioasq-passage-{pid}" for pid in relevant_passage_ids]
    answer = str(row["answer"]).strip()

    return {
        "id": case_id,
        "question": str(row["question"]).strip(),
        "reference_answer": answer,
        "reference_claims": _split_claims(answer),
        "relevant_document_ids": relevant_document_ids,
        "relevance_grades": dict.fromkeys(
            relevant_document_ids, _UNIFORM_RELEVANCE_GRADE
        ),
        "answerable": True,
        "difficulty": "single-hop" if len(relevant_document_ids) == 1 else "multi-hop",
        "split": _split_for(case_id, _TEST_SPLIT_FRACTION),
        "tags": ["bioasq"],
        "reviewed_by": None,
    }


def load_bioasq_cases(
    qa_parquet_path: Path,
    *,
    split: Split | None = None,
) -> tuple[EvalCase, ...]:
    if not qa_parquet_path.exists():
        raise FileNotFoundError(
            f"{qa_parquet_path} not found; run `uv run poe eval-download-corpus` first",
        )

    rows = pq.read_table(qa_parquet_path).to_pylist()
    cases = tuple(
        sorted(
            (case_from_dict(_row_to_case_dict(row)) for row in rows),
            key=lambda case: case.id,
        ),
    )
    if split is not None:
        cases = tuple(case for case in cases if case.split == split)
    return cases


def dataset_provenance(qa_parquet_path: Path) -> dict[str, object]:
    content_hash = "sha256:" + hashlib.sha256(qa_parquet_path.read_bytes()).hexdigest()
    return {
        "version": "bioasq-v1",
        "content_hash": content_hash,
        "corpus_id": "rag-mini-bioasq",
        "source": _SOURCE_URL,
        "source_license": _SOURCE_LICENSE,
    }
