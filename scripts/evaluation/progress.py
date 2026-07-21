from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalProgressStats:
    """Running aggregates for the eval-run tqdm postfix."""

    completed: int = 0
    sum_recall_at_k: float = 0.0
    sum_hit_rate_at_k: float = 0.0
    sum_ndcg_at_k: float = 0.0
    sum_latency_ms: float = 0.0
    sum_ask_latency_ms: float = 0.0
    ask_latency_count: int = 0
    sum_faithfulness: float = 0.0
    faithfulness_count: int = 0
    judge_error_cases: int = 0

    def observe(self, record: dict[str, object]) -> None:
        self.completed += 1
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            self.sum_recall_at_k += float(metrics.get("recall_at_k") or 0.0)
            self.sum_hit_rate_at_k += float(metrics.get("hit_rate_at_k") or 0.0)
            self.sum_ndcg_at_k += float(metrics.get("ndcg_at_k") or 0.0)
        self.sum_latency_ms += float(record.get("latency_ms") or 0.0)

        ask_latency = record.get("ask_latency_ms")
        if ask_latency is not None:
            self.sum_ask_latency_ms += float(ask_latency)
            self.ask_latency_count += 1

        generation_metrics = record.get("generation_metrics")
        if isinstance(generation_metrics, dict):
            faithfulness = generation_metrics.get("faithfulness")
            if faithfulness is not None:
                self.sum_faithfulness += float(faithfulness)
                self.faithfulness_count += 1

        judge_errors = record.get("judge_errors")
        if isinstance(judge_errors, list) and judge_errors:
            self.judge_error_cases += 1

    def postfix(self) -> dict[str, object]:
        n = max(self.completed, 1)
        values: dict[str, object] = {
            "R@k": f"{self.sum_recall_at_k / n:.2f}",
            "hit": f"{self.sum_hit_rate_at_k / n:.2f}",
            "nDCG": f"{self.sum_ndcg_at_k / n:.2f}",
            "ms": f"{self.sum_latency_ms / n:.0f}",
        }
        if self.ask_latency_count:
            values["ask_ms"] = (
                f"{self.sum_ask_latency_ms / self.ask_latency_count:.0f}"
            )
        if self.faithfulness_count:
            values["faith"] = (
                f"{self.sum_faithfulness / self.faithfulness_count:.2f}"
            )
        if self.judge_error_cases:
            values["jerr"] = self.judge_error_cases
        return values
