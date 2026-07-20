from pathlib import Path

import pytest
from scripts.evaluation.manifest import DatasetManifest, load, save


def _sample_manifest(dataset: str = "bioasq") -> DatasetManifest:
    return DatasetManifest(
        dataset=dataset,
        created_at="2026-07-20T00:00:00+00:00",
        org_id=999_000,
        user_id=999_001,
        corpus_limit=None,
        corpus_fraction=None,
        raw_dir=f"datasets/{dataset}/raw",
        processed_dir=f"datasets/{dataset}/processed",
        source="https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq",
        source_license="cc-by-2.5",
        qa_content_hash="sha256:abc",
        corpus_content_hash="sha256:def",
        passages_available=2,
        passages_indexed=2,
        passages_failed=[],
        id_map={"111": "42", "222": "43"},
    )


def test_save_and_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _sample_manifest()

    written_path = save(manifest, path=path)
    loaded = load("bioasq", path=path)

    assert written_path == path
    assert loaded == manifest


def test_save_preserves_other_datasets_already_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    first = _sample_manifest("bioasq")
    second = _sample_manifest("other-dataset")

    save(first, path=path)
    save(second, path=path)

    assert load("bioasq", path=path) == first
    assert load("other-dataset", path=path) == second


def test_load_raises_a_clear_error_when_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval-prepare"):
        load("bioasq", path=tmp_path / "missing.json")


def test_load_raises_a_clear_error_when_dataset_entry_is_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    save(_sample_manifest("bioasq"), path=path)

    with pytest.raises(KeyError, match="other-dataset"):
        load("other-dataset", path=path)
