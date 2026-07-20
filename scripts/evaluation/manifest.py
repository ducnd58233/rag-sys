from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "datasets" / "manifest.json"


def dataset_dir(dataset: str) -> Path:
    return _REPO_ROOT / "datasets" / dataset


def progress_path(dataset: str) -> Path:
    return dataset_dir(dataset) / ".index-progress.jsonl"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset: str
    created_at: str
    org_id: int
    user_id: int
    corpus_limit: int | None
    corpus_fraction: float | None
    raw_dir: str
    processed_dir: str
    source: str
    source_license: str
    qa_content_hash: str
    corpus_content_hash: str
    passages_available: int
    passages_indexed: int
    passages_failed: list[str]
    # passage id -> our document_id, captured at ingest time (see datasets/README.md).
    id_map: dict[str, str]


def load(dataset: str, *, path: Path = DEFAULT_MANIFEST_PATH) -> DatasetManifest:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run poe eval-prepare` first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload.get(dataset)
    if entry is None:
        raise KeyError(
            f"{path} has no entry for dataset {dataset!r}; run "
            f"`uv run poe eval-prepare --dataset {dataset}` first",
        )
    return DatasetManifest(**entry)


def save(manifest: DatasetManifest, *, path: Path = DEFAULT_MANIFEST_PATH) -> Path:
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload[manifest.dataset] = asdict(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
