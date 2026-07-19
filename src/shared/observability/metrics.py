from __future__ import annotations

from opentelemetry import metrics

_meter = metrics.get_meter("rag-sys")

gen_ai_client_operation_duration = _meter.create_histogram(
    "gen_ai.client.operation.duration",
    unit="s",
)
gen_ai_client_token_usage = _meter.create_histogram(
    "gen_ai.client.token.usage",
    unit="{token}",
)
generation_ask_duration = _meter.create_histogram(
    "rag.generation.ask.duration",
    unit="s",
)
generation_ask_responses = _meter.create_counter(
    "rag.generation.ask.responses",
    unit="{response}",
)
generation_ask_citations = _meter.create_histogram(
    "rag.generation.ask.citations",
    unit="{citation}",
)
generation_query_analysis_duration = _meter.create_histogram(
    "rag.generation.query_analysis.duration",
    unit="s",
)
generation_retrieval_query_duration = _meter.create_histogram(
    "rag.generation.retrieval_query.duration",
    unit="s",
)
generation_context_merge_duration = _meter.create_histogram(
    "rag.generation.context_merge.duration",
    unit="s",
)
generation_ask_subquestions = _meter.create_histogram(
    "rag.generation.ask.subquestions",
    unit="{question}",
)
generation_ask_retrieval_queries = _meter.create_histogram(
    "rag.generation.ask.retrieval_queries",
    unit="{query}",
)
generation_ask_contexts = _meter.create_histogram(
    "rag.generation.ask.contexts",
    unit="{context}",
)
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
messaging_dlq_count = _meter.create_counter(
    "rag.messaging.dlq",
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
