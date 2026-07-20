"""Retrieval-only evaluation entry point (docs/rag-evaluation/SPEC.md FR-EVAL-2, FR-EVAL-9,
FR-EVAL-11, T2-02). Loads the golden dataset, retrieves against the running app over HTTP
(ADR-EV-001), scores every case with scripts/evaluation/metrics/retrieval.py, and writes a
run directory under runs/evaluation/<MM-DD-YYYY>-<slug>/.

Requires the app running (`uv run app`) with the fixture corpus already indexed
(`uv run poe eval-index-corpus`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scripts.evaluation.client import EvalHttpClient, RetrievalResponse
from scripts.evaluation.config import DEFAULT_EVAL_ORG_ID
from scripts.evaluation.dataset import EvalCase, load_dataset
from scripts.evaluation.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scripts.evaluation.report import render_summary

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "datasets" / "golden" / "golden-v0.1.jsonl"
_RUNS_DIR = _REPO_ROOT / "runs" / "evaluation"

_ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "dusty",
    "eager",
    "faint",
    "glossy",
    "hollow",
    "ivory",
    "jolly",
    "keen",
    "misty",
    "noble",
    "quiet",
    "rapid",
    "silent",
    "tidy",
    "vivid",
)
_NOUNS = (
    "falcon",
    "comet",
    "cedar",
    "harbor",
    "lantern",
    "meadow",
    "otter",
    "pebble",
    "quartz",
    "raven",
    "summit",
    "thicket",
    "umber",
    "violet",
    "willow",
    "zephyr",
)


def _run_slug() -> str:
    adjective = random.choice(
        _ADJECTIVES
    )  # noqa: S311 - a friendly label, not a security token
    noun = random.choice(_NOUNS)  # noqa: S311
    suffix = secrets.token_hex(2)
    return f"{adjective}-{noun}-{suffix}"


def _score_case(
    case: EvalCase,
    response: RetrievalResponse,
    *,
    top_k: int,
) -> dict[str, object]:
    relevant = set(case.relevant_document_ids)
    retrieved_filenames = [item.filename or item.document_id for item in response.items]

    return {
        "case_id": case.id,
        "split": case.split,
        "difficulty": case.difficulty,
        "tags": list(case.tags),
        "answerable": case.answerable,
        "query": case.question,
        "retrieved_filenames": retrieved_filenames,
        "scores": [item.score for item in response.items],
        "router_kind": response.router_kind,
        "latency_ms": response.latency_ms,
        "metrics": {
            "recall_at_k": recall_at_k(retrieved_filenames, relevant, top_k),
            "precision_at_k": precision_at_k(retrieved_filenames, relevant, top_k),
            "hit_rate_at_k": hit_rate_at_k(retrieved_filenames, relevant, top_k),
            "reciprocal_rank": reciprocal_rank(retrieved_filenames, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved_filenames, case.relevance_grades, top_k),
        },
    }


async def _run_once(
    cases: list[EvalCase],
    *,
    base_url: str,
    org_id: int,
    top_k: int,
    concurrency: int,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _score(client: EvalHttpClient, case: EvalCase) -> dict[str, object]:
        async with semaphore:
            response = await client.retrieve(
                org_id=org_id, query=case.question, top_k=top_k
            )
        return _score_case(case, response, top_k=top_k)

    async with EvalHttpClient(base_url) as client:
        return await asyncio.gather(*(_score(client, case) for case in cases))


async def run_evaluation(
    *,
    dataset_path: Path,
    split: str | None,
    base_url: str,
    org_id: int,
    top_k: int,
    runs: int,
    concurrency: int,
) -> Path:
    cases = list(load_dataset(dataset_path, split=split))  # type: ignore[arg-type]
    if not cases:
        raise ValueError(f"no cases found for split={split!r} in {dataset_path}")

    meta_path = dataset_path.parent / f"{dataset_path.stem}.meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )

    run_dir = _RUNS_DIR / f"{datetime.now().strftime('%m-%d-%Y')}-{_run_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records_by_run: list[list[dict[str, object]]] = []
    for run_index in range(1, runs + 1):
        records = await _run_once(
            cases,
            base_url=base_url,
            org_id=org_id,
            top_k=top_k,
            concurrency=concurrency,
        )
        records_by_run.append(records)
        results_path = run_dir / f"results-run-{run_index}.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"run {run_index}/{runs}: wrote {results_path}")

    config = {
        "dataset_path": str(dataset_path.relative_to(_REPO_ROOT)),
        "dataset_version": meta.get("version", dataset_path.stem),
        "dataset_content_hash": meta.get("content_hash", "unknown"),
        "corpus_id": meta.get("corpus_id", "unknown"),
        "split": split or "all",
        "case_count": len(cases),
        "base_url": base_url,
        "org_id": org_id,
        "top_k": top_k,
        "runs": runs,
        "concurrency": concurrency,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    summary = render_summary(config=config, records_by_run=records_by_run)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(f"\nwrote {run_dir / 'config.yaml'}")
    print(f"wrote {run_dir / 'summary.md'}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--org-id", type=int, default=DEFAULT_EVAL_ORG_ID)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args(argv)

    asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            split=args.split,
            base_url=args.base_url,
            org_id=args.org_id,
            top_k=args.top_k,
            runs=args.runs,
            concurrency=args.concurrency,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
