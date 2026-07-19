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
- Langfuse (profile `llm`): http://localhost:3002

Key dashboards in Grafana:

- `00 - Overview`: API rate/error/latency, LLM ask latency, token usage, worker lag, memory, producer and broker.
- `20 - Worker / Consumer`: worker step duration, consume latency, in-flight messages, lag, memory.
- `40 - Runtime / Resources`: process and container memory, CPU, GC.
- `60 - Request Traces`: recent, slow, worker, and error traces. Open a trace row to inspect the span waterfall for one request.
- `70 - LLM / RAG Ask`: `/generation/ask` latency, LLM operation latency, token usage, refusal/error rate, and citation coverage.

Langfuse is enabled by default in the local observability stack because the self-hosted core is free for
development. It uses the dev-only keys from `.env.example` unless overridden.
