from __future__ import annotations

import json
from pathlib import Path


def test_grafana_dashboards_are_valid_json() -> None:
    dashboard_dir = Path("deployments/observability/grafana/dashboards")

    dashboards = list(dashboard_dir.glob("*.json"))

    assert dashboards
    for dashboard in dashboards:
        json.loads(dashboard.read_text(encoding="utf-8"))


def test_dashboards_expose_worker_runtime_and_trace_signals() -> None:
    dashboard_dir = Path("deployments/observability/grafana/dashboards")
    rendered = "\n".join(
        dashboard.read_text(encoding="utf-8")
        for dashboard in dashboard_dir.glob("*.json")
    )

    assert "_by_sum" not in rendered
    assert "slo:messaging_consume_duration:p95" in rendered
    assert "rag_messaging_consume_inflight" in rendered
    assert "process_memory_usage_bytes" in rendered
    assert "process_runtime_memory_bytes" in rendered
    assert '"type": "traces"' in rendered
    assert 'duration > 500ms' in rendered
