# Observability stack

Collect backends ship with the **base** Docker Compose stack
(`deployments/docker/docker-compose.yml`): OTEL Collector → Tempo + Prometheus,
plus Kafka JMX and cAdvisor scrapes. Config files and Grafana assets stay under
this directory.

| Layer | Where | What it does |
|-------|--------|--------------|
| Collect | `make docker-up` (base compose) | Persist metrics/traces in named volumes |
| Dashboards / UIs | `make docker-up-ui` | Kibana, Kafka UI, Grafana |
| LLM analytics | `make obs-up-llm` | Langfuse + optional collector export overlay |

## Running

```bash
make docker-up       # app infra + OTEL collect (recommended default)
make docker-up-ui    # Kibana + Kafka UI + Grafana (history already in volumes)
make obs-up-llm      # Langfuse (+ OTEL export to Langfuse)
make obs-up-all      # base + all UIs + Langfuse
```

```bash
make docker-down-ui  # stop Kibana / Kafka UI / Grafana; keep collecting
make obs-down-llm    # stop Langfuse; collector falls back to Tempo/Prom only
make docker-down     # stop base (+ UI if it was composed in)
make obs-down-all    # stop base + UI + Langfuse
```

App processes send OTLP to `http://localhost:4317` (`OTEL_EXPORTER_OTLP_ENDPOINT`).
With `make docker-up`, spans land in Tempo and metrics in Prometheus even if Grafana
is stopped. Start `make docker-up-ui` later to browse that history.

Langfuse only stores LLM analytics while its stack is up (collector started with
`docker-compose.observability.langfuse-export.yml` via `obs-up-llm` / `obs-up-all`).
Tempo/Prometheus history is **not** backfilled into Langfuse when you turn it on later.

## Endpoints

- Grafana: http://localhost:3001 (`make docker-up-ui`)
- Prometheus: http://localhost:9090 (`make docker-up`)
- Tempo: http://localhost:3200 (`make docker-up`)
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

Langfuse uses the dev-only keys from `.env.example` unless overridden. Prefer
`make docker-up` day-to-day and only start Langfuse when you need its UI.
