# Observability stack

Telemetry is split so you can **collect** metrics/traces continuously (cheap relative to
Grafana/Langfuse UIs) and open dashboards later against the same Prometheus/Tempo volumes.

| Layer | Compose file(s) | What it does |
|-------|-----------------|--------------|
| Collect | `docker-compose.observability.yml` | OTEL Collector → Tempo + Prometheus; JMX + cAdvisor scrapes |
| Dashboards | `docker-compose.observability.ui.yml` | Grafana only (reads Prometheus/Tempo) |
| LLM analytics | `docker-compose.langfuse.yml` + `docker-compose.observability.langfuse-export.yml` | Langfuse + optional second collector config that also exports traces to Langfuse |

## Running

```bash
make obs-up          # collect only (recommended default while developing)
make obs-up-ui       # Grafana on top of collect (historical data already in volumes)
make obs-up-core     # collect + Grafana (no Langfuse)
make obs-up-llm      # collect + Langfuse (+ OTEL export to Langfuse)
make obs-up-all      # collect + Grafana + Langfuse
```

```bash
make obs-down-ui     # stop Grafana only; keep collecting
make obs-down-llm    # stop Langfuse; collector falls back to Tempo/Prom only
make obs-down-core   # stop collect + Grafana
make obs-down        # stop collect + Grafana + Langfuse
make obs-down-all    # same as obs-down
```

App processes still send OTLP to `http://localhost:4317` (`OTEL_EXPORTER_OTLP_ENDPOINT`).
As long as `make obs-up` is running, spans land in Tempo and metrics in Prometheus even if
Grafana is stopped. Start `make obs-up-ui` later to browse that history.

Langfuse only stores LLM analytics while its stack is up (and the collector is started with
the langfuse-export overlay via `obs-up-llm` / `obs-up-all`). Tempo/Prometheus history is
**not** backfilled into Langfuse when you turn Langfuse on later.

## Endpoints

- Grafana: http://localhost:3001 (`make obs-up-ui` or `obs-up-core` / `obs-up-all`)
- Prometheus: http://localhost:9090 (`make obs-up`)
- Tempo: http://localhost:3200 (`make obs-up`)
- Collector metrics: http://localhost:8889/metrics
- cAdvisor: http://localhost:8081
- Kafka JMX exporter: http://localhost:5556/metrics
- Langfuse LLM analytics: http://localhost:3002 (`make obs-up-llm` / `obs-up-all`)

Key dashboards in Grafana:

- `00 - Overview`: API rate/error/latency, LLM ask latency, token usage, worker lag, memory, producer and broker.
- `20 - Worker / Consumer`: worker step duration, consume latency, in-flight messages, lag, memory.
- `40 - Runtime / Resources`: process and container memory, CPU, GC.
- `60 - Request Traces`: recent, slow, worker, and error traces. Open a trace row to inspect the span waterfall for one request.
- `70 - LLM / RAG Ask`: `/generation/ask` latency, LLM operation latency, token usage, refusal/error rate, citation coverage, retrieval query fan-out, context counts, router fallback ratio, retrieval strategy latency, retrieval strategy result counts, and simple vs decomposed strategy split.

For complex RAG requests, inspect Tempo traces for these spans:

- `generation.ask`
- `generation.query_analysis`
- `generation.retrieve.original`
- `generation.retrieve.subquestion`
- `retrieval.retrieve`
- `retrieval.strategy.<name>`
- `generation.context_merge`
- `generation.answer_synthesis`

Langfuse uses the dev-only keys from `.env.example` unless overridden. Prefer `make obs-up`
day-to-day and only start Langfuse when you need its UI.
