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
from scripts.evaluation.client import AskResponse, EvalHttpClient, RetrievalResponse
from scripts.evaluation.dataset import EvalCase
from scripts.evaluation.judge.config import (
    JudgeConfig,
    build_judge_chat_model,
    resolve_judge_config,
)
from scripts.evaluation.judge.rubrics import RUBRIC_VERSION
from scripts.evaluation.judge.runner import (
    decompose_claims,
    judge_citation_support,
    judge_claim_correctness,
    judge_faithfulness,
    judge_relevancy,
)
from scripts.evaluation.manifest import DEFAULT_MANIFEST_PATH, DatasetManifest
from scripts.evaluation.manifest import load as load_manifest
from scripts.evaluation.manifest import progress_path
from scripts.evaluation.metrics import citation, generation
from scripts.evaluation.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scripts.evaluation.progress import EvalProgressStats
from scripts.evaluation.report import judge_error_rate, render_summary
from scripts.evaluation.sources import SOURCES, get_source

from src.shared.app.ports import IChatModel
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import ChatSettings, LoggingSettings
from tqdm import tqdm

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _REPO_ROOT / "runs" / "evaluation"
_DEFAULT_DATASET = "bioasq"
_LIMIT_SAMPLE_SEED = 20260720
_DEFAULT_TIMEOUT_SECONDS = ChatSettings().timeout_seconds
_JUDGE_ERROR_RATE_THRESHOLD = 0.05

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


_EMPTY_GENERATION_METRICS: dict[str, float | None] = {
    "faithfulness": None,
    "answer_relevancy": None,
    "claim_precision": None,
    "claim_recall": None,
    "claim_f1": None,
    "completeness": None,
}
_EMPTY_CITATION_METRICS: dict[str, float | None] = {
    "citation_precision": None,
    "citation_recall": None,
}


async def _score_generation(
    judge_chat: IChatModel,
    case: EvalCase,
    ask_response: AskResponse,
) -> dict[str, object]:
    """Runs claim decomposition + the judge once per response and derives every
    generation/citation metric from that single set of judgments - never re-decomposes per metric.
    """
    record: dict[str, object] = {
        "answer": ask_response.answer,
        "refused": ask_response.refused,
        "citation_count": len(ask_response.citations),
        "ask_latency_ms": ask_response.latency_ms,
        "claims": [],
        "unsupported_claim_indices": [],
        "generated_claim_labels": [],
        "covered_reference_indices": [],
        "generation_metrics": dict(_EMPTY_GENERATION_METRICS),
        "citation_metrics": dict(_EMPTY_CITATION_METRICS),
        "judge_errors": [],
    }
    if ask_response.refused or not ask_response.answer.strip():
        return record

    judge_errors: list[str] = []
    claims_result = await decompose_claims(judge_chat, ask_response.answer)
    if claims_result.judge_error:
        judge_errors.append(claims_result.judge_error)
    claims = claims_result.claims
    context = "\n\n".join(
        citation_item.content for citation_item in ask_response.citations
    )

    faithfulness_result = await judge_faithfulness(
        judge_chat, claims=claims, context=context
    )
    relevancy_result = await judge_relevancy(
        judge_chat, question=case.question, answer=ask_response.answer
    )
    correctness_result = await judge_claim_correctness(
        judge_chat,
        generated_claims=claims,
        reference_claims=case.reference_claims,
    )
    claim_text = " ".join(claims)
    citation_pairs = (
        tuple((claim_text, item.content) for item in ask_response.citations)
        if claims
        else ()
    )
    citation_result = await judge_citation_support(judge_chat, citation_pairs)

    for judged in (
        faithfulness_result,
        relevancy_result,
        correctness_result,
        citation_result,
    ):
        if judged.judge_error:
            judge_errors.append(judged.judge_error)

    supported_claim_count = len(claims) - len(
        faithfulness_result.unsupported_claim_indices
    )
    # A metric whose underlying judge call failed must stay None, never silently
    # compute from empty/default judgment data - each metric below is gated on its own judge_error,
    # and citation_recall additionally on faithfulness_result's, since it reuses
    # supported_claim_count.
    faithfulness_score = (
        None
        if faithfulness_result.judge_error
        else generation.faithfulness(
            claim_count=len(claims),
            unsupported_count=len(faithfulness_result.unsupported_claim_indices),
        )
    )
    relevancy_score = (
        None
        if relevancy_result.judge_error
        else generation.answer_relevancy(relevancy_result.score)
    )
    precision = (
        None
        if correctness_result.judge_error
        else generation.claim_precision(correctness_result.generated_claim_labels)
    )
    recall = (
        None
        if correctness_result.judge_error
        else generation.claim_recall(
            reference_claim_count=len(case.reference_claims),
            covered_reference_count=len(correctness_result.covered_reference_indices),
        )
    )
    completeness_score = (
        None
        if correctness_result.judge_error
        else generation.completeness(
            reference_claim_count=len(case.reference_claims),
            covered_reference_count=len(correctness_result.covered_reference_indices),
        )
    )
    citation_precision_score = (
        None
        if citation_result.judge_error
        else citation.citation_precision(
            cited_count=len(ask_response.citations),
            supported_count=len(citation_result.supported_pair_indices),
        )
    )
    citation_recall_score = (
        None
        if (citation_result.judge_error or faithfulness_result.judge_error)
        else citation.citation_recall(
            claims_requiring_evidence_count=len(claims),
            claims_with_valid_citation_count=supported_claim_count,
        )
    )

    record["claims"] = list(claims)
    record["unsupported_claim_indices"] = list(
        faithfulness_result.unsupported_claim_indices
    )
    # FR-EVAL-3: per-claim labels stored on the raw result, not just folded into
    # the aggregate score, so a disagreement can be inspected rather than argued
    # about.
    record["generated_claim_labels"] = list(correctness_result.generated_claim_labels)
    record["covered_reference_indices"] = list(
        correctness_result.covered_reference_indices
    )
    record["generation_metrics"] = {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "claim_precision": precision,
        "claim_recall": recall,
        "claim_f1": generation.claim_f1(precision, recall),
        "completeness": completeness_score,
    }
    record["citation_metrics"] = {
        "citation_precision": citation_precision_score,
        "citation_recall": citation_recall_score,
    }
    record["judge_errors"] = judge_errors
    return record


async def _run_once(
    cases: list[EvalCase],
    *,
    base_url: str,
    org_id: int,
    top_k: int,
    concurrency: int,
    timeout_seconds: float,
    judge_chat: IChatModel | None,
    run_label: str,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)
    stats = EvalProgressStats()
    progress = tqdm(total=len(cases), desc=run_label, unit="case")

    async def _score(client: EvalHttpClient, case: EvalCase) -> dict[str, object]:
        async with semaphore:
            response = await client.retrieve(
                org_id=org_id, query=case.question, top_k=top_k
            )
            record = _score_case(case, response, top_k=top_k)
            if judge_chat is not None:
                ask_response = await client.ask(
                    org_id=org_id, query=case.question, top_k=top_k
                )
                record.update(await _score_generation(judge_chat, case, ask_response))
        stats.observe(record)
        progress.set_postfix(**stats.postfix(), refresh=False)
        progress.update(1)
        return record

    try:
        async with EvalHttpClient(base_url, timeout_seconds=timeout_seconds) as client:
            return await asyncio.gather(*(_score(client, case) for case in cases))
    finally:
        progress.close()

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
    score_generation: bool = True,
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

    judge_config: JudgeConfig | None = None
    judge_chat: IChatModel | None = None
    if score_generation:
        judge_config = resolve_judge_config()
        judge_chat = build_judge_chat_model(judge_config)
        if judge_config.self_preference_risk:
            logger.warning(
                "judge model (%s/%s) is the same as the app's generation model - "
                "scores carry a self-preference risk (PLAN.md R2). Set "
                "EVAL_JUDGE_PROVIDER/EVAL_JUDGE_MODEL to use a different judge.",
                judge_config.provider,
                judge_config.model,
            )

    records_by_run: list[list[dict[str, object]]] = []
    for run_index in range(1, runs + 1):
        records = await _run_once(
            cases,
            base_url=base_url,
            org_id=resolved_org_id,
            top_k=top_k,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            judge_chat=judge_chat,
            run_label=f"eval {run_index}/{runs}",
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
        "score_generation": score_generation,
        "judge": (
            {
                "provider": judge_config.provider,
                "model": judge_config.model,
                "temperature": judge_config.temperature,
                "rubric_version": RUBRIC_VERSION,
                "self_preference_risk": judge_config.self_preference_risk,
            }
            if judge_config is not None
            else None
        ),
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

    error_rate = judge_error_rate(records_by_run)
    if error_rate is not None and error_rate["rate"] > _JUDGE_ERROR_RATE_THRESHOLD:
        raise RuntimeError(
            f"judge error rate {error_rate['rate']:.2%} exceeds the "
            f"{_JUDGE_ERROR_RATE_THRESHOLD:.0%} threshold (FR-EVAL-6) - "
            f"{error_rate['errored']}/{error_rate['n']} cases had a judge_error. "
            f"Artefacts were still written to {run_dir} for inspection."
        )

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
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help=(
            "score retrieval only, skipping /generation/ask + the LLM judge "
            "(faithfulness, relevancy, claim F1, completeness, citation "
            "precision/recall). Generation scoring runs by default and adds one "
            "HTTP call plus up to five judge calls per case - use this for a fast "
            "retrieval-only iteration loop. Judge model/provider default to the "
            "app's own generation model unless EVAL_JUDGE_PROVIDER/EVAL_JUDGE_MODEL "
            "are set (see PLAN.md R2 - self-preference risk when they match)."
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
            score_generation=not args.skip_generation,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
