from __future__ import annotations

from scripts.evaluation.progress import EvalProgressStats


def test_eval_progress_stats_postfix_averages_retrieval_metrics() -> None:
    stats = EvalProgressStats()
    stats.observe(
        {
            "latency_ms": 100.0,
            "metrics": {
                "recall_at_k": 1.0,
                "hit_rate_at_k": 1.0,
                "ndcg_at_k": 0.5,
            },
        }
    )
    stats.observe(
        {
            "latency_ms": 300.0,
            "metrics": {
                "recall_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "ndcg_at_k": 0.5,
            },
        }
    )

    postfix = stats.postfix()

    assert stats.completed == 2
    assert postfix["R@k"] == "0.50"
    assert postfix["hit"] == "0.50"
    assert postfix["nDCG"] == "0.50"
    assert postfix["ms"] == "200"


def test_eval_progress_stats_includes_generation_fields_when_present() -> None:
    stats = EvalProgressStats()
    stats.observe(
        {
            "latency_ms": 50.0,
            "ask_latency_ms": 120.0,
            "metrics": {
                "recall_at_k": 1.0,
                "hit_rate_at_k": 1.0,
                "ndcg_at_k": 1.0,
            },
            "generation_metrics": {"faithfulness": 0.8},
            "judge_errors": ["timeout"],
        }
    )

    postfix = stats.postfix()

    assert postfix["ask_ms"] == "120"
    assert postfix["faith"] == "0.80"
    assert postfix["jerr"] == 1
