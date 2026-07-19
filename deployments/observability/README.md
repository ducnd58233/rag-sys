# Observability stack

## Running

```bash
make obs-up          # observability stack only
make docker-up       # app infra + observability together
make obs-up-llm      # also start Langfuse (opt-in)
```

```bash
make obs-down
make obs-down-llm
```

## Dashboards

- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090
- Tempo: http://localhost:3200
- Collector metrics: http://localhost:8889/metrics
- cAdvisor: http://localhost:8081
- Kafka JMX exporter: http://localhost:5556/metrics
- Langfuse (profile `llm`): http://localhost:3002
