import pytest
from scripts.evaluation.report import aggregate_across_runs, render_summary


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
