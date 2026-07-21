import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.evaluation.sources import bioasq

_CORPUS_ROWS = [
    {"id": "1", "passage": "Passage one text."},
    {"id": "2", "passage": "Passage two text."},
    {"id": "3", "passage": "Passage three text."},
    {"id": "4", "passage": "Passage four text."},
    {"id": "5", "passage": "Passage five text."},
]

_QA_ROWS = [
    {
        "id": "q1",
        "question": "What is passage three about?",
        "answer": "It is about topic A.",
        "relevant_passage_ids": json.dumps(["3"]),
    },
    {
        "id": "q2",
        "question": "What connects passages one and two?",
        "answer": "They share topic B. They were published together.",
        "relevant_passage_ids": json.dumps(["1", "2"]),
    },
]


def _write_corpus(raw_dir: Path, rows: list[dict[str, object]] = _CORPUS_ROWS) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), bioasq.corpus_path(raw_dir))


def _write_qa(raw_dir: Path, rows: list[dict[str, object]] = _QA_ROWS) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), bioasq.qa_path(raw_dir))


def test_processed_filename_is_stable_per_passage_id() -> None:
    assert bioasq.processed_filename("123") == "bioasq-passage-123.md"


def test_iter_corpus_passages_without_limit_returns_every_passage_sorted_by_id(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)

    passages = bioasq.iter_corpus_passages(tmp_path)

    assert [p.id for p in passages] == ["1", "2", "3", "4", "5"]
    assert passages[0].text == "Passage one text."


def test_iter_corpus_passages_raises_when_corpus_not_downloaded(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="eval-prepare"):
        bioasq.iter_corpus_passages(tmp_path)


def test_iter_corpus_passages_with_limit_prioritizes_smallest_qa_case_first(
    tmp_path: Path,
) -> None:
    """q1 needs only passage 3 (a set of size 1); q2 needs passages 1 and 2 (size 2).

    The packing rule sorts cases by relevant-set size ascending, so with a budget of
    only 1 passage, q1's single passage must be selected over either of q2's pair.
    """
    _write_corpus(tmp_path)
    _write_qa(tmp_path)

    passages = bioasq.iter_corpus_passages(tmp_path, limit=1)

    assert [p.id for p in passages] == ["3"]


def test_iter_corpus_passages_with_limit_fills_remaining_budget_in_id_order(
    tmp_path: Path,
) -> None:
    """budget of 4 fully covers both QA cases (1 + 2 = 3 passages) with one slot left
    over; the leftover slot is filled from the untouched passages in id order (4)."""
    _write_corpus(tmp_path)
    _write_qa(tmp_path)

    passages = bioasq.iter_corpus_passages(tmp_path, limit=4)

    assert sorted(p.id for p in passages) == ["1", "2", "3", "4"]


def test_iter_corpus_passages_with_fraction_computes_a_rounded_target(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    _write_qa(tmp_path)

    # 5 passages * 0.4 = 2, rounded.
    passages = bioasq.iter_corpus_passages(tmp_path, fraction=0.4)

    assert len(passages) == 2


def test_load_qa_cases_maps_relevant_passage_ids_through_id_map(
    tmp_path: Path,
) -> None:
    _write_qa(tmp_path)
    id_map = {"1": "doc-1", "2": "doc-2", "3": "doc-3"}

    cases = bioasq.load_qa_cases(tmp_path, id_map)

    by_id = {case.id: case for case in cases}
    assert set(by_id) == {"bioasq-qa-q1", "bioasq-qa-q2"}
    assert by_id["bioasq-qa-q1"].relevant_document_ids == ("doc-3",)
    assert by_id["bioasq-qa-q1"].difficulty == "single-hop"
    assert by_id["bioasq-qa-q2"].relevant_document_ids == ("doc-1", "doc-2")
    assert by_id["bioasq-qa-q2"].difficulty == "multi-hop"


def test_load_qa_cases_sets_uniform_relevance_grade_and_bioasq_tag(
    tmp_path: Path,
) -> None:
    _write_qa(tmp_path)
    id_map = {"1": "doc-1", "2": "doc-2", "3": "doc-3"}

    cases = bioasq.load_qa_cases(tmp_path, id_map)

    for case in cases:
        assert case.answerable is True
        assert case.tags == ("bioasq",)
        assert case.reviewed_by is None
        assert all(grade == 1 for grade in case.relevance_grades.values())
        assert set(case.relevance_grades) == set(case.relevant_document_ids)


def test_load_qa_cases_splits_a_multi_sentence_answer_into_multiple_claims(
    tmp_path: Path,
) -> None:
    _write_qa(tmp_path)
    id_map = {"1": "doc-1", "2": "doc-2", "3": "doc-3"}

    cases = bioasq.load_qa_cases(tmp_path, id_map)

    q2 = next(case for case in cases if case.id == "bioasq-qa-q2")
    assert q2.reference_claims == (
        "They share topic B.",
        "They were published together.",
    )


def test_load_qa_cases_drops_cases_whose_relevant_passages_were_never_ingested(
    tmp_path: Path,
) -> None:
    _write_qa(tmp_path)
    # Passage "3" (q1's only relevant passage) was never ingested.
    id_map = {"1": "doc-1", "2": "doc-2"}

    cases = bioasq.load_qa_cases(tmp_path, id_map)

    assert [case.id for case in cases] == ["bioasq-qa-q2"]


def test_load_qa_cases_raises_when_qa_parquet_not_downloaded(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval-prepare"):
        bioasq.load_qa_cases(tmp_path, {})


def test_load_qa_cases_split_assignment_is_deterministic_across_calls(
    tmp_path: Path,
) -> None:
    _write_qa(tmp_path)
    id_map = {"1": "doc-1", "2": "doc-2", "3": "doc-3"}

    first = {case.id: case.split for case in bioasq.load_qa_cases(tmp_path, id_map)}
    second = {case.id: case.split for case in bioasq.load_qa_cases(tmp_path, id_map)}

    assert first == second
    assert set(first.values()) <= {"dev", "test"}


def test_load_qa_cases_split_filter_returns_only_matching_cases(
    tmp_path: Path,
) -> None:
    _write_qa(tmp_path)
    id_map = {"1": "doc-1", "2": "doc-2", "3": "doc-3"}

    all_cases = bioasq.load_qa_cases(tmp_path, id_map)
    dev_cases = bioasq.load_qa_cases(tmp_path, id_map, split="dev")

    expected_ids = {case.id for case in all_cases if case.split == "dev"}
    assert {case.id for case in dev_cases} == expected_ids


def test_provenance_reports_source_and_content_hashes(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    _write_qa(tmp_path)

    provenance = bioasq.provenance(tmp_path)

    expected_qa_hash = (
        "sha256:"
        + hashlib.sha256(
            bioasq.qa_path(tmp_path).read_bytes(),
        ).hexdigest()
    )
    expected_corpus_hash = (
        "sha256:"
        + hashlib.sha256(
            bioasq.corpus_path(tmp_path).read_bytes(),
        ).hexdigest()
    )

    assert provenance["corpus_id"] == "bioasq"
    assert provenance["source_license"] == "cc-by-2.5"
    assert provenance["qa_content_hash"] == expected_qa_hash
    assert provenance["corpus_content_hash"] == expected_corpus_hash
