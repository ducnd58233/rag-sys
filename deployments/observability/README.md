# Observability stack

## Running

```bash
make obs-up          # observability stack + Langfuse for local LLM analytics
make docker-up       # app infra + observability + Langfuse together
make obs-up-core     # Prometheus/Grafana/Tempo only, no Langfuse
```

```bash
make obs-down
make obs-down-core
```

## Dashboards

- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090
- Tempo: http://localhost:3200
- Collector metrics: http://localhost:8889/metrics
- cAdvisor: http://localhost:8081
- Kafka JMX exporter: http://localhost:5556/metrics
- Langfuse LLM analytics (profile `llm`): http://localhost:3002

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

Langfuse is enabled by default in the local observability stack because the self-hosted OSS core is free for
development. It uses the dev-only keys from `.env.example` unless overridden. Use `make obs-up-core` if you only need
Grafana/Prometheus/Tempo and want to skip the Langfuse ClickHouse dependency.
