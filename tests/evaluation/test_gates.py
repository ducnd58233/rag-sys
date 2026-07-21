from pathlib import Path

from scripts.evaluation.gates import (
    GateSpec,
    evaluate_gate,
    evaluate_gates,
    flatten_summary_metrics,
    load_baseline,
    render_delta_table,
)

_MAX_DROP = GateSpec("recall", "recall_at_10", "max_drop_pp", 0.03, "reason")
_MAX_VALUE = GateSpec(
    "unsupported", "unsupported_answer_rate", "max_value", 0.08, "reason"
)
_NO_REGRESSION = GateSpec("f1", "claim_f1", "no_regression", 0.0, "reason")


def test_max_drop_pp_passes_within_threshold() -> None:
    result = evaluate_gate(
        _MAX_DROP,
        candidate={"recall_at_10": 0.68},
        baseline={"recall_at_10": 0.70},
    )
    assert result.passed is True


def test_max_drop_pp_fails_beyond_threshold() -> None:
    result = evaluate_gate(
        _MAX_DROP,
        candidate={"recall_at_10": 0.60},
        baseline={"recall_at_10": 0.70},
    )
    assert result.passed is False


def test_max_drop_pp_passes_when_candidate_improves() -> None:
    result = evaluate_gate(
        _MAX_DROP,
        candidate={"recall_at_10": 0.85},
        baseline={"recall_at_10": 0.70},
    )
    assert result.passed is True


def test_max_drop_pp_passes_without_a_baseline_yet() -> None:
    result = evaluate_gate(_MAX_DROP, candidate={"recall_at_10": 0.5}, baseline={})
    assert result.passed is True


def test_max_value_fails_above_threshold() -> None:
    result = evaluate_gate(
        _MAX_VALUE,
        candidate={"unsupported_answer_rate": 0.10},
        baseline={},
    )
    assert result.passed is False


def test_max_value_passes_at_or_below_threshold() -> None:
    result = evaluate_gate(
        _MAX_VALUE,
        candidate={"unsupported_answer_rate": 0.08},
        baseline={},
    )
    assert result.passed is True


def test_no_regression_fails_when_candidate_is_lower_than_baseline() -> None:
    result = evaluate_gate(
        _NO_REGRESSION,
        candidate={"claim_f1": 0.5},
        baseline={"claim_f1": 0.6},
    )
    assert result.passed is False


def test_no_regression_passes_when_candidate_matches_or_exceeds_baseline() -> None:
    result = evaluate_gate(
        _NO_REGRESSION,
        candidate={"claim_f1": 0.6},
        baseline={"claim_f1": 0.6},
    )
    assert result.passed is True


def test_gate_passes_by_omission_when_candidate_has_no_value() -> None:
    result = evaluate_gate(_MAX_VALUE, candidate={}, baseline={})
    assert result.passed is True
    assert result.candidate_value is None


def test_evaluate_gates_runs_every_spec() -> None:
    results = evaluate_gates(
        candidate={"recall_at_10": 0.9, "unsupported_answer_rate": 0.01},
        baseline={},
        gates=(_MAX_DROP, _MAX_VALUE),
    )
    assert [r.name for r in results] == ["recall", "unsupported"]
    assert all(r.passed for r in results)


def test_render_delta_table_flags_failures() -> None:
    results = evaluate_gates(
        candidate={"recall_at_10": 0.1},
        baseline={"recall_at_10": 0.9},
        gates=(_MAX_DROP,),
    )
    table = render_delta_table(results)
    assert "NO" in table
    assert "recall" in table


def test_load_baseline_returns_empty_dict_for_missing_file(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "missing.json") == {}


def test_flatten_summary_metrics_pulls_from_records() -> None:
    records_by_run = [
        [
            {
                "metrics": {
                    "recall_at_k": 0.7,
                    "precision_at_k": None,
                    "hit_rate_at_k": None,
                    "reciprocal_rank": None,
                    "ndcg_at_k": None,
                },
                "latency_ms": 50.0,
                "answerable": True,
                "refused": False,
                "generation_metrics": {
                    "faithfulness": 0.9,
                    "answer_relevancy": None,
                    "claim_precision": None,
                    "claim_recall": None,
                    "claim_f1": 0.8,
                    "completeness": None,
                },
                "citation_metrics": {
                    "citation_precision": None,
                    "citation_recall": None,
                },
                "ask_latency_ms": 500.0,
                "judge_errors": [],
            },
        ],
    ]
    flat = flatten_summary_metrics(records_by_run)
    assert flat["recall_at_10"] == 0.7
    assert flat["faithfulness"] == 0.9
    assert flat["claim_f1"] == 0.8
    assert flat["retrieval_p95_ms"] == 50.0
    assert flat["e2e_p95_ms"] == 500.0
