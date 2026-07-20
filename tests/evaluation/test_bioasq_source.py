from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.evaluation.bioasq_source import dataset_provenance, load_bioasq_cases

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_QA_PARQUET = (
    _REPO_ROOT / "datasets" / "bioasq" / "raw" / "question-answer-passages.parquet"
)

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


def test_load_bioasq_cases_maps_rows_onto_the_eval_case_schema(tmp_path: Path) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")

    cases = load_bioasq_cases(qa_parquet)

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


def test_load_bioasq_cases_filters_by_split(tmp_path: Path) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")

    all_cases = load_bioasq_cases(qa_parquet)
    dev_cases = load_bioasq_cases(qa_parquet, split="dev")
    test_cases = load_bioasq_cases(qa_parquet, split="test")

    assert len(dev_cases) + len(test_cases) == len(all_cases)
    assert all(case.split == "dev" for case in dev_cases)
    assert all(case.split == "test" for case in test_cases)


def test_load_bioasq_cases_is_deterministic_across_calls(tmp_path: Path) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")

    first = load_bioasq_cases(qa_parquet)
    second = load_bioasq_cases(qa_parquet)

    assert first == second


def test_load_bioasq_cases_raises_a_clear_error_when_parquet_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="eval-download-corpus"):
        load_bioasq_cases(tmp_path / "missing.parquet")


def test_dataset_provenance_hashes_the_raw_parquet_bytes(tmp_path: Path) -> None:
    qa_parquet = _write_qa_parquet(tmp_path / "question-answer-passages.parquet")

    provenance = dataset_provenance(qa_parquet)

    assert provenance["version"] == "bioasq-v1"
    assert provenance["corpus_id"] == "rag-mini-bioasq"
    assert provenance["content_hash"].startswith("sha256:")

    # Changing the file must change the hash - this is what pins config.yaml's
    # dataset_content_hash to the actual bytes a run scored against.
    qa_parquet.write_bytes(qa_parquet.read_bytes() + b"\x00")
    assert dataset_provenance(qa_parquet)["content_hash"] != provenance["content_hash"]


@pytest.mark.skipif(
    not _REAL_QA_PARQUET.exists(),
    reason=(
        "datasets/bioasq/raw/question-answer-passages.parquet is not committed - run "
        "`uv run poe eval-download-corpus` first"
    ),
)
def test_real_bioasq_qa_parquet_loads_all_4719_answerable_cases() -> None:
    cases = load_bioasq_cases(_REAL_QA_PARQUET)

    assert len(cases) == 4719
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.answerable for case in cases)
    assert all(
        set(case.relevance_grades) == set(case.relevant_document_ids) for case in cases
    )
