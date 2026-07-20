import hashlib
import json
from pathlib import Path

import pytest
from scripts.evaluation.dataset import DatasetValidationError, load_dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]

_VALID_ANSWERABLE_CASE = {
    "id": "case-001",
    "question": "What is the model dimension of the base Transformer?",
    "reference_answer": "512.",
    "reference_claims": ["The base Transformer's model dimension is 512."],
    "relevant_document_ids": ["paper-a.pdf"],
    "relevance_grades": {"paper-a.pdf": 3},
    "answerable": True,
    "difficulty": "single-hop",
    "split": "dev",
    "tags": ["transformer"],
    "reviewed_by": None,
}

_VALID_UNANSWERABLE_CASE = {
    "id": "case-002",
    "question": "Who approved deployment deploy-9921?",
    "reference_answer": None,
    "reference_claims": [],
    "relevant_document_ids": [],
    "relevance_grades": {},
    "answerable": False,
    "expected_behavior": "abstain",
    "difficulty": "unanswerable",
    "split": "test",
    "tags": ["unanswerable"],
    "reviewed_by": None,
}


def _write_dataset(tmp_path: Path, *cases: dict[str, object]) -> Path:
    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    return path


def test_valid_answerable_and_unanswerable_cases_load(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, _VALID_ANSWERABLE_CASE, _VALID_UNANSWERABLE_CASE)
    cases = load_dataset(path)
    assert [case.id for case in cases] == ["case-001", "case-002"]
    assert cases[0].answerable is True
    assert cases[1].answerable is False
    assert cases[1].expected_behavior == "abstain"


def test_split_filter_returns_only_matching_cases(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, _VALID_ANSWERABLE_CASE, _VALID_UNANSWERABLE_CASE)
    dev_cases = load_dataset(path, split="dev")
    assert [case.id for case in dev_cases] == ["case-001"]


@pytest.mark.parametrize(
    "override",
    [
        {"reference_answer": "should be null"},
        {"reference_claims": ["should be empty"]},
        {"relevant_document_ids": ["should-be-empty.pdf"]},
        {"expected_behavior": "answer"},
    ],
)
def test_unanswerable_case_rejects_inconsistent_fields(
    tmp_path: Path,
    override: dict[str, object],
) -> None:
    broken = {**_VALID_UNANSWERABLE_CASE, **override}
    path = _write_dataset(tmp_path, broken)
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


@pytest.mark.parametrize(
    "override",
    [
        {"reference_claims": []},
        {"relevant_document_ids": []},
    ],
)
def test_answerable_case_rejects_empty_required_fields(
    tmp_path: Path,
    override: dict[str, object],
) -> None:
    broken = {**_VALID_ANSWERABLE_CASE, **override}
    path = _write_dataset(tmp_path, broken)
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_relevance_grades_key_must_be_subset_of_relevant_document_ids(
    tmp_path: Path,
) -> None:
    broken = {
        **_VALID_ANSWERABLE_CASE,
        "relevance_grades": {"paper-a.pdf": 3, "not-in-relevant-ids.pdf": 2},
    }
    path = _write_dataset(tmp_path, broken)
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    broken = {**_VALID_ANSWERABLE_CASE, "split": "staging"}
    path = _write_dataset(tmp_path, broken)
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_malformed_json_line_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_built_golden_dataset_loads_and_matches_its_content_hash() -> None:
    dataset_path = _REPO_ROOT / "datasets" / "golden" / "bioasq-v1.jsonl"
    meta_path = _REPO_ROOT / "datasets" / "golden" / "bioasq-v1.meta.json"
    if not dataset_path.exists():
        pytest.skip(
            "bioasq-v1.jsonl is not committed (~6MB, exceeds the large-file hook) - "
            "run `uv run poe eval-download-corpus && uv run poe eval-build-golden` first",
        )

    cases = load_dataset(dataset_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert len(cases) == meta["case_count"]
    actual_hash = "sha256:" + hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert actual_hash == meta["content_hash"]
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.answerable for case in cases)
    assert all(
        set(case.relevance_grades) == set(case.relevant_document_ids) for case in cases
    )
