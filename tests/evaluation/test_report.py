import pytest
from scripts.evaluation.report import (
    aggregate_abstention_across_runs,
    aggregate_across_runs,
    aggregate_generation_across_runs,
    breakdown_by_strategy_combination,
    judge_error_rate,
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


def _generation_record(
    *,
    answerable: bool,
    refused: bool,
    faithfulness: float | None = None,
    judge_errors: list[str] | None = None,
) -> dict:
    return {
        "answerable": answerable,
        "refused": refused,
        "judge_errors": judge_errors or [],
        "generation_metrics": {
            "faithfulness": faithfulness,
            "answer_relevancy": None,
            "claim_precision": None,
            "claim_recall": None,
            "claim_f1": None,
            "completeness": None,
        },
        "citation_metrics": {"citation_precision": None, "citation_recall": None},
    }


def test_aggregate_generation_across_runs_reports_mean_of_per_run_means() -> None:
    records_by_run = [
        [_generation_record(answerable=True, refused=False, faithfulness=0.8)],
        [_generation_record(answerable=True, refused=False, faithfulness=0.6)],
    ]
    aggregates = aggregate_generation_across_runs(records_by_run)
    assert aggregates["faithfulness"]["mean"] == pytest.approx(0.7)
    assert aggregates["faithfulness"]["runs"] == 2


def test_aggregate_generation_across_runs_is_empty_without_generation_metrics_key() -> (
    None
):
    records_by_run = [[{"metrics": {"recall_at_k": 0.5}}]]
    assert aggregate_generation_across_runs(records_by_run) == {}


def test_aggregate_abstention_pools_confusion_matrix_across_runs() -> None:
    records_by_run = [
        [
            _generation_record(answerable=True, refused=False),  # TP
            _generation_record(answerable=False, refused=True),  # TN
        ],
        [
            _generation_record(answerable=True, refused=True),  # FN
            _generation_record(answerable=False, refused=False),  # FP
        ],
    ]
    abstention = aggregate_abstention_across_runs(records_by_run)
    assert abstention is not None
    assert abstention["n"] == 4
    assert abstention["confusion_matrix"] == {
        "true_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "false_positive": 1,
    }
    assert abstention["abstention_precision"] == pytest.approx(0.5)
    assert abstention["abstention_recall"] == pytest.approx(0.5)


def test_aggregate_abstention_is_none_for_retrieval_only_records() -> None:
    records_by_run = [[{"metrics": {"recall_at_k": 0.5}, "answerable": True}]]
    assert aggregate_abstention_across_runs(records_by_run) is None


def test_judge_error_rate_counts_records_with_any_judge_error() -> None:
    records_by_run = [
        [
            _generation_record(answerable=True, refused=False, judge_errors=["boom"]),
            _generation_record(answerable=True, refused=False, judge_errors=[]),
        ],
    ]
    result = judge_error_rate(records_by_run)
    assert result == {"rate": pytest.approx(0.5), "errored": 1, "n": 2}


def test_judge_error_rate_is_none_without_generation_scoring() -> None:
    records_by_run = [[{"metrics": {"recall_at_k": 0.5}}]]
    assert judge_error_rate(records_by_run) is None


def test_render_summary_includes_judge_and_abstention_sections() -> None:
    records_by_run = [
        [
            {
                **_generation_record(
                    answerable=True,
                    refused=False,
                    faithfulness=0.9,
                ),
                "strategies": ["hybrid"],
                "metrics": {
                    "recall_at_k": 0.6,
                    "precision_at_k": None,
                    "hit_rate_at_k": None,
                    "reciprocal_rank": None,
                    "ndcg_at_k": None,
                },
                "latency_ms": 10.0,
            },
        ],
    ]
    summary = render_summary(
        config={
            "dataset_version": "bioasq",
            "dataset_content_hash": "sha256:abc",
            "corpus_id": "bioasq",
            "split": "dev",
            "org_id": 999000,
            "top_k": 10,
            "base_url": "http://localhost:8000",
            "runs": 1,
            "generated_at": "2026-07-21T00:00:00+00:00",
            "judge": {
                "provider": "ollama",
                "model": "qwen3:4b-instruct",
                "rubric_version": "judge-rubrics-v1",
                "self_preference_risk": True,
            },
        },
        records_by_run=records_by_run,
    )
    assert "SELF-PREFERENCE RISK" in summary
    assert "Generation and citation metrics" in summary
    assert "Abstention (FR-EVAL-5)" in summary
    assert "Judge error rate" in summary
