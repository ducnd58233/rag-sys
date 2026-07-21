from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.evaluation.report import (
    aggregate_abstention_across_runs,
    aggregate_across_runs,
    aggregate_generation_across_runs,
)

GateDirection = Literal["max_drop_pp", "max_value", "no_regression"]


@dataclass(frozen=True, slots=True)
class GateSpec:
    name: str
    metric: str
    direction: GateDirection
    threshold: float
    reason: str


DEFAULT_GATES: tuple[GateSpec, ...] = (
    GateSpec(
        "recall_at_10",
        "recall_at_10",
        "max_drop_pp",
        0.03,
        "Recall@10 must not drop more than 3pp vs baseline (requires --top-k 10)",
    ),
    GateSpec(
        "faithfulness",
        "faithfulness",
        "max_drop_pp",
        0.02,
        "Faithfulness must not drop more than 2pp vs baseline",
    ),
    GateSpec(
        "unsupported_answer_rate",
        "unsupported_answer_rate",
        "max_value",
        0.08,
        "Unsupported-answer rate must not exceed 8%",
    ),
    GateSpec(
        "claim_f1",
        "claim_f1",
        "no_regression",
        0.0,
        "Answer (claim) F1 must not fall below baseline",
    ),
    GateSpec(
        "retrieval_p95_ms",
        "retrieval_p95_ms",
        "max_value",
        200.0,
        "Retrieval p95 latency must not exceed 200ms",
    ),
    GateSpec(
        "e2e_p95_ms",
        "e2e_p95_ms",
        "max_value",
        2500.0,
        "End-to-end (ask) p95 latency must not exceed 2.5s",
    ),
)


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    candidate_value: float | None
    baseline_value: float | None
    reason: str


def evaluate_gate(
    spec: GateSpec,
    *,
    candidate: dict[str, float | None],
    baseline: dict[str, float | None],
) -> GateResult:
    candidate_value = candidate.get(spec.metric)
    baseline_value = baseline.get(spec.metric)
    if candidate_value is None:
        # Nothing to gate on (e.g. a retrieval-only run scoring a generation
        # metric) - passes by omission rather than failing on missing data.
        return GateResult(spec.name, True, None, baseline_value, spec.reason)

    if spec.direction == "max_value":
        passed = candidate_value <= spec.threshold
    elif spec.direction == "max_drop_pp":
        passed = (
            baseline_value is None
            or (baseline_value - candidate_value) <= spec.threshold
        )
    elif spec.direction == "no_regression":
        passed = (
            baseline_value is None or candidate_value >= baseline_value - spec.threshold
        )
    else:  # pragma: no cover - exhaustive over GateDirection
        raise ValueError(f"unknown gate direction: {spec.direction}")

    return GateResult(spec.name, passed, candidate_value, baseline_value, spec.reason)


def evaluate_gates(
    *,
    candidate: dict[str, float | None],
    baseline: dict[str, float | None],
    gates: Sequence[GateSpec] = DEFAULT_GATES,
) -> list[GateResult]:
    return [
        evaluate_gate(spec, candidate=candidate, baseline=baseline) for spec in gates
    ]


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render_delta_table(results: Sequence[GateResult]) -> str:
    lines = [
        "| Gate | Candidate | Baseline | Passed | Reason |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {_fmt(result.candidate_value)} | "
            f"{_fmt(result.baseline_value)} | {'yes' if result.passed else 'NO'} | "
            f"{result.reason} |",
        )
    return "\n".join(lines) + "\n"


def _p95(sorted_values: Sequence[float]) -> float | None:
    if not sorted_values:
        return None
    return sorted_values[int(len(sorted_values) * 0.95) - 1]


def flatten_summary_metrics(
    records_by_run: Sequence[Sequence[dict[str, object]]],
) -> dict[str, float | None]:
    """Reduces a run's per-case records to the flat metric names DEFAULT_GATES
    compares. `recall_at_10` is only meaningful when the run used --top-k 10 -
    this function does not check that, since it has no access to the run's
    config; callers gating on it must ensure that themselves."""
    retrieval = aggregate_across_runs(records_by_run)
    generation = aggregate_generation_across_runs(records_by_run)
    abstention = aggregate_abstention_across_runs(records_by_run)
    retrieval_latencies = sorted(
        float(record["latency_ms"])  # type: ignore[arg-type]
        for records in records_by_run
        for record in records
        if "latency_ms" in record
    )
    ask_latencies = sorted(
        float(record["ask_latency_ms"])  # type: ignore[arg-type]
        for records in records_by_run
        for record in records
        if "ask_latency_ms" in record
    )
    return {
        "recall_at_10": (retrieval.get("recall_at_k") or {}).get("mean"),
        "faithfulness": (generation.get("faithfulness") or {}).get("mean"),
        "claim_f1": (generation.get("claim_f1") or {}).get("mean"),
        "unsupported_answer_rate": (
            abstention["unsupported_answer_rate"] if abstention else None
        ),
        "retrieval_p95_ms": _p95(retrieval_latencies),
        "e2e_p95_ms": _p95(ask_latencies),
    }


def load_baseline(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a run directory's aggregated metrics against a committed "
            "baseline.json and fail on regression (FR-EVAL-12)."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("baseline.json"))
    args = parser.parse_args(argv)

    run_files = sorted(args.run_dir.glob("results-run-*.jsonl"))
    if not run_files:
        parser.error(f"no results-run-*.jsonl files found under {args.run_dir}")
    records_by_run = [
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for path in run_files
    ]

    candidate = flatten_summary_metrics(records_by_run)
    baseline = load_baseline(args.baseline)
    results = evaluate_gates(candidate=candidate, baseline=baseline)
    print(render_delta_table(results))

    if not all(result.passed for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
