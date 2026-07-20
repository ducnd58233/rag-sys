from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
_QA_PARQUET = (
    _REPO_ROOT / "datasets" / "bioasq" / "raw" / "question-answer-passages.parquet"
)
_OUTPUT_PATH = _REPO_ROOT / "datasets" / "golden" / "bioasq-v1.jsonl"
_META_PATH = _REPO_ROOT / "datasets" / "golden" / "bioasq-v1.meta.json"

_TEST_SPLIT_FRACTION = 0.2
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SOURCE_URL = "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq"


def _split_claims(answer: str) -> list[str]:
    sentences = [
        s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()
    ]
    return sentences or [answer.strip()]


def _split_for(case_id: str, fraction: float) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < fraction else "dev"


def _to_case(row: dict[str, object]) -> dict[str, object]:
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
        "relevance_grades": dict.fromkeys(relevant_document_ids, 1),
        "answerable": True,
        "difficulty": "single-hop" if len(relevant_document_ids) == 1 else "multi-hop",
        "split": _split_for(case_id, _TEST_SPLIT_FRACTION),
        "tags": ["bioasq"],
        "reviewed_by": None,
    }


def build_golden(
    *,
    qa_parquet_path: Path = _QA_PARQUET,
    output_path: Path = _OUTPUT_PATH,
    meta_path: Path = _META_PATH,
) -> tuple[Path, Path, int]:
    if not qa_parquet_path.exists():
        raise FileNotFoundError(
            f"{qa_parquet_path} not found; run `uv run poe eval-download-corpus` first",
        )

    rows = pq.read_table(qa_parquet_path).to_pylist()
    cases = sorted((_to_case(row) for row in rows), key=lambda case: case["id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")

    difficulty_distribution: dict[str, int] = {}
    split_distribution: dict[str, int] = {}
    for case in cases:
        difficulty_distribution[case["difficulty"]] = (
            difficulty_distribution.get(case["difficulty"], 0) + 1
        )
        split_distribution[case["split"]] = split_distribution.get(case["split"], 0) + 1

    content_hash = "sha256:" + hashlib.sha256(output_path.read_bytes()).hexdigest()
    meta = {
        "version": "bioasq-v1",
        "content_hash": content_hash,
        "case_count": len(cases),
        "corpus_id": "rag-mini-bioasq",
        "source": _SOURCE_URL,
        "source_license": "cc-by-2.5",
        "reviewed": False,
        "fact_checked": False,
        "provenance": (
            "question, reference_answer, and relevant_document_ids are taken verbatim from "
            "the question-answer-passages split of rag-datasets/rag-mini-bioasq (itself "
            "derived from the official BioASQ Task 11b training set). No question or answer "
            "here was generated or edited by this script - every case_count row in the "
            "source split is represented exactly once."
        ),
        "relevance_grades_note": (
            "relevance_grades is a uniform 1 for every relevant id: BioASQ ships binary "
            "relevance judgments, not the 0-3 graded scale this harness's ndcg_at_k was "
            "designed around, so nDCG on this dataset is a binary-relevance variant, not a "
            "true multi-grade signal."
        ),
        "known_gap": (
            "Every case has answerable=true. The source QA split contains no unanswerable "
            "questions, so this dataset alone cannot exercise abstention behavior. No "
            "unanswerable cases were fabricated to fill that gap - see datasets/README.md."
        ),
        "difficulty_distribution": difficulty_distribution,
        "split_distribution": split_distribution,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return output_path, meta_path, len(cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-parquet", type=Path, default=_QA_PARQUET)
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH)
    parser.add_argument("--meta-output", type=Path, default=_META_PATH)
    args = parser.parse_args(argv)

    output_path, meta_path, case_count = build_golden(
        qa_parquet_path=args.qa_parquet,
        output_path=args.output,
        meta_path=args.meta_output,
    )
    print(f"wrote {output_path} ({case_count} cases)")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
