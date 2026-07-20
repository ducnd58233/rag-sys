import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.evaluation.build_bioasq_golden import build_golden
from scripts.evaluation.dataset import load_dataset

_ROWS = [
    {
        "id": 0,
        "question": "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
        "answer": "It is a multifactorial disorder. Multiple loci are involved.",
        "relevant_passage_ids": "[111, 222]",
    },
    {
        "id": 1,
        "question": "Is the protein Papilin secreted?",
        "answer": "Yes, papilin is a secreted protein",
        "relevant_passage_ids": "[333]",
    },
]


def _write_qa_parquet(path: Path) -> Path:
    table = pa.table(
        {
            "question": [row["question"] for row in _ROWS],
            "answer": [row["answer"] for row in _ROWS],
            "relevant_passage_ids": [row["relevant_passage_ids"] for row in _ROWS],
            "id": [row["id"] for row in _ROWS],
        },
    )
    pq.write_table(table, path)
    return path


def test_build_golden_maps_bioasq_rows_onto_the_eval_case_schema(
    tmp_path: Path,
) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")
    output_path = tmp_path / "bioasq-v1.jsonl"
    meta_path = tmp_path / "bioasq-v1.meta.json"

    build_golden(
        qa_parquet_path=qa_parquet, output_path=output_path, meta_path=meta_path
    )
    cases = load_dataset(output_path)

    assert [case.id for case in cases] == ["bioasq-qa-0", "bioasq-qa-1"]

    multi_hop, single_hop = cases
    assert multi_hop.difficulty == "multi-hop"
    assert multi_hop.relevant_document_ids == (
        "bioasq-passage-111",
        "bioasq-passage-222",
    )
    assert multi_hop.relevance_grades == {
        "bioasq-passage-111": 1,
        "bioasq-passage-222": 1,
    }
    assert multi_hop.reference_claims == (
        "It is a multifactorial disorder.",
        "Multiple loci are involved.",
    )
    assert multi_hop.answerable is True

    assert single_hop.difficulty == "single-hop"
    assert single_hop.relevant_document_ids == ("bioasq-passage-333",)


def test_build_golden_writes_a_meta_file_matching_the_output(tmp_path: Path) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")
    output_path = tmp_path / "bioasq-v1.jsonl"
    meta_path = tmp_path / "bioasq-v1.meta.json"

    build_golden(
        qa_parquet_path=qa_parquet, output_path=output_path, meta_path=meta_path
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["case_count"] == 2
    assert meta["content_hash"].startswith("sha256:")
    assert sum(meta["difficulty_distribution"].values()) == 2
    assert sum(meta["split_distribution"].values()) == 2


def test_build_golden_is_deterministic_across_runs(tmp_path: Path) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")

    _, _, _ = build_golden(
        qa_parquet_path=qa_parquet,
        output_path=tmp_path / "run1.jsonl",
        meta_path=tmp_path / "run1.meta.json",
    )
    _, _, _ = build_golden(
        qa_parquet_path=qa_parquet,
        output_path=tmp_path / "run2.jsonl",
        meta_path=tmp_path / "run2.meta.json",
    )

    assert (tmp_path / "run1.jsonl").read_bytes() == (
        tmp_path / "run2.jsonl"
    ).read_bytes()
