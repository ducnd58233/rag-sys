from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scripts.evaluation.client import EvalHttpClient, RetrievalResponse
from scripts.evaluation.dataset import EvalCase
from scripts.evaluation.manifest import DEFAULT_MANIFEST_PATH, DatasetManifest
from scripts.evaluation.manifest import load as load_manifest
from scripts.evaluation.manifest import progress_path
from scripts.evaluation.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scripts.evaluation.report import render_summary
from scripts.evaluation.sources import SOURCES, get_source

from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import ChatSettings, LoggingSettings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _REPO_ROOT / "runs" / "evaluation"
_DEFAULT_DATASET = "bioasq"
_LIMIT_SAMPLE_SEED = 20260720
_DEFAULT_TIMEOUT_SECONDS = ChatSettings().timeout_seconds

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
    # Real DB document_id, not metadata.filename - see datasets/README.md.
    retrieved_ids = [item.document_id for item in response.items]

    return {
        "case_id": case.id,
        "split": case.split,
        "difficulty": case.difficulty,
        "tags": list(case.tags),
        "answerable": case.answerable,
        "query": case.question,
        "retrieved_document_ids": retrieved_ids,
        "scores": [item.score for item in response.items],
        "router_kind": response.router_kind,
        "router_reason": response.router_reason,
        "strategies": list(response.strategies),
        "latency_ms": response.latency_ms,
        "metrics": {
            "recall_at_k": recall_at_k(retrieved_ids, relevant, top_k),
            "precision_at_k": precision_at_k(retrieved_ids, relevant, top_k),
            "hit_rate_at_k": hit_rate_at_k(retrieved_ids, relevant, top_k),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, case.relevance_grades, top_k),
        },
    }


async def _run_once(
    cases: list[EvalCase],
    *,
    base_url: str,
    org_id: int,
    top_k: int,
    concurrency: int,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _score(client: EvalHttpClient, case: EvalCase) -> dict[str, object]:
        async with semaphore:
            response = await client.retrieve(
                org_id=org_id, query=case.question, top_k=top_k
            )
        return _score_case(case, response, top_k=top_k)

    async with EvalHttpClient(base_url, timeout_seconds=timeout_seconds) as client:
        return await asyncio.gather(*(_score(client, case) for case in cases))


def _cleanup(manifest: DatasetManifest) -> None:
    processed_dir = _REPO_ROOT / manifest.processed_dir
    progress_file = progress_path(manifest.dataset)

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        logger.info("removed %s", processed_dir)
    if progress_file.exists():
        progress_file.unlink()
        logger.info("removed %s", progress_file)
    logger.info(
        "raw/%s and manifest.json were kept - re-run eval-prepare without --no-resume "
        "to reuse the same document_id mapping instead of re-ingesting from scratch. "
        "DB records are not deleted (no delete-document use case exists yet); drop the "
        "local stack's volumes for a fully clean slate.",
        manifest.dataset,
    )


async def run_evaluation(
    *,
    dataset: str,
    manifest_path: Path,
    split: str | None,
    base_url: str,
    org_id: int | None,
    top_k: int,
    runs: int,
    concurrency: int,
    timeout_seconds: float,
    limit: int | None = None,
    cleanup: bool = False,
) -> Path:
    source = get_source(dataset)
    manifest = load_manifest(dataset, path=manifest_path)
    resolved_org_id = org_id if org_id is not None else manifest.org_id

    cases = list(
        source.load_qa_cases(
            _REPO_ROOT / manifest.raw_dir,
            manifest.id_map,
            split=split,  # type: ignore[arg-type]
        ),
    )
    if not cases:
        raise ValueError(f"no cases found for split={split!r} in {manifest_path}")
    if limit is not None and limit < len(cases):
        # Fixed seed: a --limit run samples the same cases every time, not a fresh draw.
        cases = random.Random(_LIMIT_SAMPLE_SEED).sample(cases, limit)

    logger.info(
        "scoring %d cases (dataset=%s, split=%s, org_id=%d, top_k=%d, runs=%d)",
        len(cases),
        dataset,
        split or "all",
        resolved_org_id,
        top_k,
        runs,
    )

    run_dir = _RUNS_DIR / f"{datetime.now().strftime('%m-%d-%Y')}-{_run_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records_by_run: list[list[dict[str, object]]] = []
    for run_index in range(1, runs + 1):
        records = await _run_once(
            cases,
            base_url=base_url,
            org_id=resolved_org_id,
            top_k=top_k,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
        )
        records_by_run.append(records)
        results_path = run_dir / f"results-run-{run_index}.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        logger.info("run %d/%d: wrote %s", run_index, runs, results_path)

    config = {
        "dataset": dataset,
        "manifest_path": str(manifest_path.relative_to(_REPO_ROOT)),
        "dataset_version": manifest.corpus_content_hash,
        "dataset_content_hash": manifest.qa_content_hash,
        "corpus_id": dataset,
        "corpus_limit": manifest.corpus_limit,
        "corpus_fraction": manifest.corpus_fraction,
        "source": manifest.source,
        "source_license": manifest.source_license,
        "split": split or "all",
        "case_count": len(cases),
        "limit": limit,
        "base_url": base_url,
        "org_id": resolved_org_id,
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

    logger.info("wrote %s", run_dir / "config.yaml")
    logger.info("wrote %s", run_dir / "summary.md")

    if cleanup:
        _cleanup(manifest)

    return run_dir


def main(argv: list[str] | None = None) -> int:
    configure_logging(
        LoggingSettings(handlers=["console"]),
        service_name="evaluation-run",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=_DEFAULT_DATASET, choices=sorted(SOURCES))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="path to manifest.json (default: datasets/manifest.json)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="'all' scores every case the manifest's id_map can cover, ignoring the dev/test split",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--org-id",
        type=int,
        default=None,
        help="override the org_id to query (default: the one eval-prepare ingested into)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help=(
            "per-request HTTP timeout. A local CPU-bound chat model (query routing, "
            "complex-RAG iterations) can easily take longer than aiohttp's own 30s "
            "default, especially on a cold Ollama model load - raise this rather than "
            "treating a slow-but-eventually-successful response as a hang."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "sample at most N cases from the (already split-filtered) dataset for a fast "
            "local run, using a fixed seed for reproducibility. Omit for the full benchmark."
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "after writing results, delete the processed corpus files and the ingest "
            "progress file (both are cheaply regenerable via eval-prepare); does not "
            "touch raw downloads, manifest.json, or anything in the database"
        ),
    )
    args = parser.parse_args(argv)

    asyncio.run(
        run_evaluation(
            dataset=args.dataset,
            manifest_path=args.manifest or DEFAULT_MANIFEST_PATH,
            split=None if args.split == "all" else args.split,
            base_url=args.base_url,
            org_id=args.org_id,
            top_k=args.top_k,
            runs=args.runs,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            cleanup=args.cleanup,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
