from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import yaml
from scripts.evaluation.metrics.abstention import (
    abstention_f1,
    abstention_precision,
    abstention_recall,
    confusion_matrix,
    unsupported_answer_rate,
)

_METRIC_NAMES = (
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
)
_GENERATION_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "claim_precision",
    "claim_recall",
    "claim_f1",
    "completeness",
)
_CITATION_METRIC_NAMES = ("citation_precision", "citation_recall")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _metric_values(
    records: Sequence[dict[str, object]],
    metric_name: str,
    *,
    metrics_key: str = "metrics",
) -> list[float]:
    return [
        record[metrics_key][metric_name]  # type: ignore[index]
        for record in records
        if metrics_key in record
        and record[metrics_key].get(metric_name) is not None  # type: ignore[union-attr]
    ]


def _aggregate_metric_group(
    records_by_run: Sequence[Sequence[dict[str, object]]],
    *,
    metric_names: Sequence[str],
    metrics_key: str,
) -> dict[str, dict[str, float | int]]:
    """mean/stdev of the per-run mean, plus the contributing case count and run count.

    Reporting the mean of per-run means (rather than pooling every case from every run
    into one list) keeps case-level and run-level variance separable.
    `n` is the number of cases that actually had a defined value for this metric (a
    case with empty `relevant_document_ids`, e.g. an unanswerable case, is undefined for
    recall/precision/MRR/nDCG and is excluded, not counted as evidence for the mean).
    """
    aggregates: dict[str, dict[str, float | int]] = {}

    for metric_name in metric_names:
        per_run_means: list[float] = []
        contributing_n = 0
        for records in records_by_run:
            values = _metric_values(records, metric_name, metrics_key=metrics_key)
            if not values:
                continue
            per_run_means.append(statistics.fmean(values))
            contributing_n = max(contributing_n, len(values))
        if not per_run_means:
            continue
        aggregates[metric_name] = {
            "mean": statistics.fmean(per_run_means),
            "stdev": (
                statistics.pstdev(per_run_means) if len(per_run_means) > 1 else 0.0
            ),
            "n": contributing_n,
            "runs": len(per_run_means),
        }
    return aggregates


def aggregate_across_runs(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, dict[str, float | int]]:
    return _aggregate_metric_group(
        records_by_run,
        metric_names=_METRIC_NAMES,
        metrics_key="metrics",
    )


def aggregate_generation_across_runs(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, dict[str, float | int]]:
    return _aggregate_metric_group(
        records_by_run,
        metric_names=_GENERATION_METRIC_NAMES,
        metrics_key="generation_metrics",
    )


def aggregate_citation_across_runs(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, dict[str, float | int]]:
    return _aggregate_metric_group(
        records_by_run,
        metric_names=_CITATION_METRIC_NAMES,
        metrics_key="citation_metrics",
    )


def aggregate_abstention_across_runs(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, object] | None:
    """Pools (answerable, refused) pairs across every run into one confusion
    matrix - unlike the scalar metrics above, a count-based confusion matrix has
    no natural per-run mean/stdev, so repeated runs are treated as replicated
    trials contributing to the same matrix, not averaged. None when no record
    carries a `refused` field (retrieval-only run, --skip-generation)."""
    scored = [
        record
        for records in records_by_run
        for record in records
        if "refused" in record
    ]
    if not scored:
        return None
    pairs = [(bool(record["answerable"]), bool(record["refused"])) for record in scored]
    matrix = confusion_matrix(pairs)
    precision = abstention_precision(matrix)
    recall = abstention_recall(matrix)
    return {
        "confusion_matrix": matrix,
        "n": len(scored),
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": abstention_f1(precision, recall),
        "unsupported_answer_rate": unsupported_answer_rate(matrix),
    }


def judge_error_rate(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, object] | None:
    """FR-EVAL-6: judge configuration and error rate belong in every experiment
    record; a rate above 5% invalidates the run (checked by run.py, not here)."""
    scored = [
        record
        for records in records_by_run
        for record in records
        if "judge_errors" in record
    ]
    if not scored:
        return None
    errored = sum(1 for record in scored if record["judge_errors"])
    return {"rate": errored / len(scored), "errored": errored, "n": len(scored)}


def breakdown_by_strategy_combination(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, dict[str, tuple[float, int]]]:
    """Pooled (not per-run-mean) metric averages per distinct set of strategies the
    router actually combined for a case, e.g. "hybrid" vs "hybrid+graph". Diagnostic,
    not the headline number: it pools every case from every run together rather than
    keeping run-level variance separate, because a strategy combination may not appear
    the same number of times in every run.
    """
    pooled: dict[str, dict[str, list[float]]] = {}
    for records in records_by_run:
        for record in records:
            strategies = record.get("strategies") or []
            key = "+".join(sorted(strategies)) or "unknown"  # type: ignore[arg-type]
            bucket = pooled.setdefault(key, {name: [] for name in _METRIC_NAMES})
            metrics = record["metrics"]  # type: ignore[index]
            for name in _METRIC_NAMES:
                value = metrics.get(name)  # type: ignore[union-attr]
                if value is not None:
                    bucket[name].append(value)

    return {
        key: {
            name: (statistics.fmean(values), len(values))
            for name, values in metrics.items()
            if values
        }
        for key, metrics in pooled.items()
    }


def render_summary(
    *,
    config: dict[str, object],
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> str:
    aggregates = aggregate_across_runs(records_by_run)
    latencies = [
        record["latency_ms"]  # type: ignore[misc]
        for records in records_by_run
        for record in records
    ]
    latencies_sorted = sorted(float(latency) for latency in latencies)
    p95 = (
        latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]
        if latencies_sorted
        else None
    )

    judge_config = config.get("judge")

    lines = [
        "# Evaluation run summary",
        "",
        f"- dataset: {config.get('dataset_version')} ({config.get('dataset_content_hash')})",
        f"- corpus: {config.get('corpus_id')}",
        f"- split: {config.get('split')}",
        f"- org_id: {config.get('org_id')}",
        f"- top_k: {config.get('top_k')}",
        f"- base_url: {config.get('base_url')}",
        f"- runs: {config.get('runs')}",
        f"- generated_at: {config.get('generated_at')}",
    ]
    if isinstance(judge_config, dict):
        risk_note = (
            " (SELF-PREFERENCE RISK - same as the generation model, PLAN.md R2)"
            if judge_config.get("self_preference_risk")
            else ""
        )
        lines.append(
            f"- judge: {judge_config.get('provider')}/{judge_config.get('model')} "
            f"(rubrics {judge_config.get('rubric_version')}){risk_note}",
        )
    else:
        lines.append("- judge: not scored (--skip-generation)")

    lines.extend(
        [
            "",
            "## Retrieval metrics",
            "",
            "| Metric | Mean | Stdev | n | runs |",
            "|---|---|---|---|---|",
        ],
    )
    for metric_name in _METRIC_NAMES:
        stats = aggregates.get(metric_name)
        if stats is None:
            lines.append(f"| {metric_name} | - | - | 0 | 0 |")
            continue
        lines.append(
            f"| {metric_name} | {stats['mean']:.4f} | {stats['stdev']:.4f} "
            f"| {stats['n']} | {stats['runs']} |",
        )

    lines.append("")
    lines.append(
        (
            f"Retrieval p95 latency: {p95:.1f} ms"
            if p95 is not None
            else "Retrieval p95 latency: n/a"
        ),
    )
    lines.append("")
    lines.append(
        "A single-run number is not reportable; this file always reflects the run "
        "count above.",
    )

    generation_aggregates = aggregate_generation_across_runs(records_by_run)
    citation_aggregates = aggregate_citation_across_runs(records_by_run)
    if generation_aggregates or citation_aggregates:
        lines.append("")
        lines.append("## Generation and citation metrics")
        lines.append("")
        lines.append("| Metric | Mean | Stdev | n | runs |")
        lines.append("|---|---|---|---|---|")
        for metric_name in (*_GENERATION_METRIC_NAMES, *_CITATION_METRIC_NAMES):
            stats = generation_aggregates.get(metric_name) or citation_aggregates.get(
                metric_name,
            )
            if stats is None:
                lines.append(f"| {metric_name} | - | - | 0 | 0 |")
                continue
            lines.append(
                f"| {metric_name} | {stats['mean']:.4f} | {stats['stdev']:.4f} "
                f"| {stats['n']} | {stats['runs']} |",
            )

    abstention = aggregate_abstention_across_runs(records_by_run)
    if abstention is not None:
        matrix = abstention["confusion_matrix"]
        lines.append("")
        lines.append("## Abstention (FR-EVAL-5)")
        lines.append("")
        lines.append(
            f"n={abstention['n']} (pooled across all runs, not a per-run mean) - "
            "true_positive=answered correctly, false_negative=wrongly abstained, "
            "true_negative=correctly abstained, false_positive=hallucinated answer. "
            "Safety refusals are not distinguished from evidence-abstentions here - "
            "`safety_decision` does not exist yet (see PLAN.md R1'/`datasets/README.md`).",
        )
        lines.append("")
        lines.append(
            f"- confusion matrix: {matrix}",
        )
        lines.append(
            f"- abstention precision: {_fmt(abstention['abstention_precision'])}",
        )
        lines.append(f"- abstention recall: {_fmt(abstention['abstention_recall'])}")
        lines.append(f"- abstention F1: {_fmt(abstention['abstention_f1'])}")
        lines.append(
            f"- unsupported-answer rate: {_fmt(abstention['unsupported_answer_rate'])}",
        )

    error_rate = judge_error_rate(records_by_run)
    if error_rate is not None:
        lines.append("")
        lines.append(
            f"Judge error rate: {error_rate['rate']:.4f} "
            f"({error_rate['errored']}/{error_rate['n']} cases) - "
            "a run above 5% is invalid per FR-EVAL-6 and is rejected by run.py.",
        )

    strategy_breakdown = breakdown_by_strategy_combination(records_by_run)
    if strategy_breakdown:
        lines.append("")
        lines.append(
            "## Breakdown by strategy combination (diagnostic, pooled across runs)"
        )
        lines.append("")
        lines.append(
            "| Strategies | n | recall_at_k | precision_at_k | hit_rate_at_k | reciprocal_rank | ndcg_at_k |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for key in sorted(strategy_breakdown):
            stats = strategy_breakdown[key]
            row = [key]
            n = max((count for _, count in stats.values()), default=0)
            row.append(str(n))
            for metric_name in _METRIC_NAMES:
                if metric_name in stats:
                    mean, _ = stats[metric_name]
                    row.append(f"{mean:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-render summary.md from an existing run directory.",
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    run_files = sorted(args.run_dir.glob("results-run-*.jsonl"))
    records_by_run = [
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        for path in run_files
    ]

    summary = render_summary(config=config, records_by_run=records_by_run)
    (args.run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"wrote {args.run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
