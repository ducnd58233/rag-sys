from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq
from scripts.evaluation.dataset import EvalCase, Split, case_from_dict

logger = logging.getLogger(__name__)

NAME = "bioasq"
MIME_TYPE = "text/markdown"

_SOURCE_URL = "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq"
_SOURCE_LICENSE = "cc-by-2.5"

_HF_API_ROOT = (
    "https://huggingface.co/api/datasets/rag-datasets/rag-mini-bioasq/parquet"
)
_QA_FILENAME = "question-answer-passages.parquet"
_CORPUS_FILENAME = "text-corpus.parquet"
_URLS = {
    _QA_FILENAME: f"{_HF_API_ROOT}/question-answer-passages/test/0.parquet",
    _CORPUS_FILENAME: f"{_HF_API_ROOT}/text-corpus/passages/0.parquet",
}

_USER_AGENT = "rag-sys-evaluation-corpus-downloader/1.0"
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 120.0
_PARQUET_MAGIC = b"PAR1"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TEST_SPLIT_FRACTION = 0.2

# BioASQ has binary relevance only, so ndcg_at_k here is a binary-relevance variant.
_UNIFORM_RELEVANCE_GRADE = 1


class Passage(NamedTuple):
    id: str
    text: str


def qa_path(raw_dir: Path) -> Path:
    return raw_dir / _QA_FILENAME


def corpus_path(raw_dir: Path) -> Path:
    return raw_dir / _CORPUS_FILENAME


def processed_filename(passage_id: str) -> str:
    return f"bioasq-passage-{passage_id}.md"


def _download_file(url: str, *, destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        logger.info("already downloaded, skipping: %s", destination.name)
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                payload = response.read()
            if not payload.startswith(_PARQUET_MAGIC):
                raise ValueError(
                    f"{url} did not return a parquet file (missing PAR1 magic header)",
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            logger.info(
                "downloaded %s (%s bytes)",
                destination.name,
                f"{len(payload):,}",
            )
            return destination
        except (urllib.error.URLError, ValueError) as error:
            last_error = error
            logger.warning(
                "download attempt %d/%d for %s failed: %s",
                attempt,
                _RETRY_ATTEMPTS,
                url,
                error,
            )
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Could not download {url} after {_RETRY_ATTEMPTS} attempts",
    ) from last_error


def download(raw_dir: Path, *, force: bool = False) -> dict[str, Path]:
    return {
        "qa": _download_file(
            _URLS[_QA_FILENAME],
            destination=qa_path(raw_dir),
            force=force,
        ),
        "corpus": _download_file(
            _URLS[_CORPUS_FILENAME],
            destination=corpus_path(raw_dir),
            force=force,
        ),
    }


def _pack_passages_for_case_coverage(raw_dir: Path, target: int) -> list[str]:
    """Greedily packs passage ids by QA case, smallest relevant_passage_ids set first, to maximize fully-covered cases within a target-passage budget."""
    qa_rows = pq.read_table(qa_path(raw_dir)).to_pylist()
    cases = sorted(
        (
            (
                str(row["id"]),
                sorted({str(pid) for pid in json.loads(row["relevant_passage_ids"])}),
            )
            for row in qa_rows
        ),
        key=lambda case: (len(case[1]), case[0]),
    )

    selected: list[str] = []
    selected_set: set[str] = set()
    for _case_id, relevant_ids in cases:
        new_ids = [pid for pid in relevant_ids if pid not in selected_set]
        if len(selected_set) + len(new_ids) > target:
            continue
        selected_set.update(new_ids)
        selected.extend(new_ids)
    return selected


def iter_corpus_passages(
    raw_dir: Path,
    *,
    limit: int | None = None,
    fraction: float | None = None,
) -> list[Passage]:
    path = corpus_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run poe eval-prepare` first"
        )

    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: row["id"])
    passages = [Passage(id=str(row["id"]), text=row["passage"]) for row in rows]

    if limit is None and fraction is None:
        return passages

    target = limit if limit is not None else round(len(passages) * fraction)  # type: ignore[operator]
    passage_by_id = {passage.id: passage for passage in passages}

    selected_ids = _pack_passages_for_case_coverage(raw_dir, target)
    if len(selected_ids) < target:
        # Fill any unused budget so --corpus-limit N still ingests close to N.
        selected_set = set(selected_ids)
        for passage in passages:
            if len(selected_ids) >= target:
                break
            if passage.id not in selected_set:
                selected_ids.append(passage.id)
                selected_set.add(passage.id)

    return [passage_by_id[passage_id] for passage_id in selected_ids]


def _split_claims(answer: str) -> list[str]:
    sentences = [
        s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()
    ]
    return sentences or [answer.strip()]


def _split_for(case_id: str, fraction: float) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < fraction else "dev"


def load_qa_cases(
    raw_dir: Path,
    id_map: dict[str, str],
    *,
    split: Split | None = None,
) -> tuple[EvalCase, ...]:
    path = qa_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run poe eval-prepare` first"
        )

    rows = pq.read_table(path).to_pylist()
    cases: list[EvalCase] = []
    skipped = 0
    for row in rows:
        case_id = f"bioasq-qa-{row['id']}"
        passage_ids = [str(pid) for pid in json.loads(row["relevant_passage_ids"])]
        mapped_ids = [id_map.get(pid) for pid in passage_ids]
        if any(document_id is None for document_id in mapped_ids):
            # A relevant passage was never ingested (partial-corpus prepare); drop it.
            skipped += 1
            continue

        answer = str(row["answer"]).strip()
        case_dict: dict[str, object] = {
            "id": case_id,
            "question": str(row["question"]).strip(),
            "reference_answer": answer,
            "reference_claims": _split_claims(answer),
            "relevant_document_ids": mapped_ids,
            "relevance_grades": dict.fromkeys(mapped_ids, _UNIFORM_RELEVANCE_GRADE),
            "answerable": True,
            "difficulty": "single-hop" if len(mapped_ids) == 1 else "multi-hop",
            "split": _split_for(case_id, _TEST_SPLIT_FRACTION),
            "tags": ["bioasq"],
            "reviewed_by": None,
        }
        cases.append(case_from_dict(case_dict))

    if skipped:
        logger.warning(
            "skipped %d/%d QA cases referencing passages outside the indexed corpus "
            "(expected for a --corpus-limit/--corpus-fraction prepare run, not for a "
            "full-corpus one)",
            skipped,
            len(rows),
        )

    sorted_cases = tuple(sorted(cases, key=lambda case: case.id))
    if split is not None:
        sorted_cases = tuple(case for case in sorted_cases if case.split == split)
    return sorted_cases


def provenance(raw_dir: Path) -> dict[str, object]:
    return {
        "corpus_id": NAME,
        "source": _SOURCE_URL,
        "source_license": _SOURCE_LICENSE,
        "qa_content_hash": "sha256:"
        + hashlib.sha256(qa_path(raw_dir).read_bytes()).hexdigest(),
        "corpus_content_hash": "sha256:"
        + hashlib.sha256(corpus_path(raw_dir).read_bytes()).hexdigest(),
    }
