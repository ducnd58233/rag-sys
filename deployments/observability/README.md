# Observability stack

Collector, Prometheus, Tempo, Grafana, a Kafka JMX exporter, and cAdvisor. Runs alongside the app stack
under the same `rag-sys` Compose project so the exporters can reach the app's `kafka` container by name.
Langfuse (LLM tracing) is a separate, opt-in profile.

## Start

```bash
make obs-up          # observability stack only
make docker-up       # app infra + observability together
make obs-up-llm       # also starts Langfuse (postgres, clickhouse, redis, minio, langfuse)
```

The API and worker are started separately via `uv run app` / `uv run worker` — see the repo `README.md`.

## Ports

| Service | Port | Notes |
|---|---|---|
| Grafana | `3001` | anonymous admin access for local dev |
| Prometheus | `9090` | |
| OTel Collector | `4317` / `4318` | gRPC / HTTP OTLP receivers, `OTEL_EXPORTER_OTLP_ENDPOINT` target |
| Collector Prometheus endpoint | `8889` | scraped by Prometheus, not for direct use |
| cAdvisor | `8081` | |
| Langfuse (profile `llm`) | `3002` | |

## Common queries

- API p95 latency: `slo:http_request_duration:p95`
- Ingestion step p95: `slo:ingestion_step_duration:p95`
- Consumer lag: `rag_messaging_consumer_lag`
- Error budget burn (fast window): `(1 - slo:http_availability:ratio_rate5m) / 0.005`

## Disabling telemetry

Set `OTEL_SDK_DISABLED=true` in `.env`. The app boots normally, exports nothing, and this stack is not
required to be running.

## Known-unverified pieces

- **Kafka JMX metrics.** `JMX_PORT` / `KAFKA_JMX_HOSTNAME` are set on the `kafka` service and
  `kafka-jmx-exporter` (`bitnami/jmx-exporter`) polls it on `9999`, mapped via
  `kafka/jmx-exporter-config.yml`. This has not been verified against a live `apache/kafka:4.1.0`
  broker in this delivery — check `docker logs kafka-jmx-exporter` and the `kafka-jmx` scrape target in
  Prometheus (`http://localhost:9090/targets`) after `make docker-up`.
- **`danielqsj/kafka-exporter`** was deliberately not added — its Kafka 4.x/KRaft compatibility is
  unverified, and app-side consumer lag (`rag_messaging_consumer_lag`, emitted by `ConsumerRuntime`) is
  already the authoritative source for our own consumer group.
- **Trace propagation across Kafka** (producer → consumer, one `trace_id`) is implemented via manual
  `traceparent` header inject/extract (`opentelemetry.propagate`) in the Kafka producer/consumer adapters
  and `ConsumerRuntime`, since there is no official `aiokafka` auto-instrumentation package. Verify with a
  real upload: check Tempo for one trace spanning the HTTP request, the Kafka produce, and the worker's
  consume + ingestion spans.
- **LLM token counts** (`rag.llm.tokens`) are only recorded when the underlying LangChain chat response
  exposes `usage_metadata` — this varies by provider/model. Embedding calls only get a span
  (`llm.embed`), no token count, since LangChain's `Embeddings` interface does not expose usage
  consistently.

## Langfuse (profile `llm`)

Not started by default. The Collector's `otlphttp/langfuse` exporter is always configured but only
succeeds once the `llm` profile is running; until then, export attempts fail silently (retried, then
dropped) with no effect on the app (FR-LLM-5).

`LANGFUSE_BASIC_AUTH` must be `base64(public_key:secret_key)` — for the `.env.example` dev defaults
(`pk-lf-dev` / `sk-lf-dev`), that is `cGstbGYtZGV2OnNrLWxmLWRldg==`.
