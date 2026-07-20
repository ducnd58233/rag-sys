import pytest
from scripts.evaluation.report import (
    aggregate_across_runs,
    breakdown_by_strategy_combination,
    render_summary,
)


def _records(*, run1: list[float | None], run2: list[float | None]) -> list[list[dict]]:
    def _to_records(values: list[float | None]) -> list[dict]:
        return [
            {
                "metrics": {
                    "recall_at_k": value,
                    "precision_at_k": None,
                    "hit_rate_at_k": None,
                    "reciprocal_rank": None,
                    "ndcg_at_k": None,
                },
                "latency_ms": 10.0,
            }
            for value in values
        ]

    return [_to_records(run1), _to_records(run2)]


def test_aggregate_reports_mean_of_per_run_means() -> None:
    records_by_run = _records(run1=[0.6, None], run2=[0.8, None])
    aggregates = aggregate_across_runs(records_by_run)
    assert aggregates["recall_at_k"]["mean"] == pytest.approx(0.7)
    assert aggregates["recall_at_k"]["runs"] == 2


def test_aggregate_n_reflects_contributing_cases_not_total_cases() -> None:
    records_by_run = _records(run1=[0.6, None], run2=[0.8, None])
    aggregates = aggregate_across_runs(records_by_run)
    # 2 cases per run, but only 1 has a defined recall value each run.
    assert aggregates["recall_at_k"]["n"] == 1


def test_metric_with_no_contributing_cases_is_omitted() -> None:
    records_by_run = _records(run1=[None], run2=[None])
    aggregates = aggregate_across_runs(records_by_run)
    assert "recall_at_k" not in aggregates


def test_render_summary_states_run_count_for_every_metric() -> None:
    records_by_run = _records(run1=[0.6], run2=[0.8])
    summary = render_summary(
        config={
            "dataset_version": "golden-v0.1",
            "dataset_content_hash": "sha256:abc",
            "corpus_id": "attention-lineage-v1",
            "split": "dev",
            "org_id": 999000,
            "top_k": 10,
            "base_url": "http://localhost:8000",
            "runs": 2,
            "generated_at": "2026-07-20T00:00:00+00:00",
        },
        records_by_run=records_by_run,
    )
    assert "runs: 2" in summary
    assert "single-run number is not reportable" in summary


def _record_with_strategies(strategies: list[str], recall: float | None) -> dict:
    return {
        "strategies": strategies,
        "metrics": {
            "recall_at_k": recall,
            "precision_at_k": None,
            "hit_rate_at_k": None,
            "reciprocal_rank": None,
            "ndcg_at_k": None,
        },
        "latency_ms": 10.0,
    }


def test_breakdown_by_strategy_combination_groups_by_sorted_strategy_set() -> None:
    records_by_run = [
        [
            _record_with_strategies(["hybrid"], 0.5),
            _record_with_strategies(["graph", "hybrid"], 0.9),
            _record_with_strategies(["hybrid", "graph"], 0.7),
        ],
    ]
    breakdown = breakdown_by_strategy_combination(records_by_run)
    assert set(breakdown) == {"hybrid", "graph+hybrid"}
    mean, n = breakdown["graph+hybrid"]["recall_at_k"]
    assert n == 2
    assert mean == pytest.approx(0.8)


def test_render_summary_includes_strategy_breakdown_table() -> None:
    records_by_run = [[_record_with_strategies(["lexical", "semantic"], 0.6)]]
    summary = render_summary(
        config={
            "dataset_version": "golden-v0.1",
            "dataset_content_hash": "sha256:abc",
            "corpus_id": "attention-lineage-v1",
            "split": "dev",
            "org_id": 999000,
            "top_k": 10,
            "base_url": "http://localhost:8000",
            "runs": 1,
            "generated_at": "2026-07-20T00:00:00+00:00",
        },
        records_by_run=records_by_run,
    )
    assert "Breakdown by strategy combination" in summary
    assert "lexical+semantic" in summary
