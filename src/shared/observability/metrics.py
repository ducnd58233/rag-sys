from __future__ import annotations

from opentelemetry import metrics

_meter = metrics.get_meter("rag-sys")

messaging_publish_duration = _meter.create_histogram(
    "rag.messaging.publish.duration",
    unit="s",
)
messaging_consume_duration = _meter.create_histogram(
    "rag.messaging.consume.duration",
    unit="s",
)
messaging_consume_inflight = _meter.create_gauge(
    "rag.messaging.consume.inflight",
    unit="{message}",
)
messaging_commit_count = _meter.create_counter(
    "rag.messaging.commit",
    unit="{message}",
)
messaging_consumer_lag = _meter.create_gauge(
    "rag.messaging.consumer.lag",
    unit="{message}",
)
ingestion_step_duration = _meter.create_histogram(
    "rag.ingestion.step.duration",
    unit="s",
)
ingestion_document_size = _meter.create_histogram(
    "rag.ingestion.document.size",
    unit="By",
)
llm_tokens = _meter.create_counter(
    "rag.llm.tokens",
    unit="{token}",
)
