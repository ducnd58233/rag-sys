COMPOSE_DIR := deployments/docker
COMPOSE_BASE := -f $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_UI := -f $(COMPOSE_DIR)/docker-compose.ui.yml
COMPOSE_VLLM := -f $(COMPOSE_DIR)/docker-compose.vllm.yml
COMPOSE_OLLAMA := -f $(COMPOSE_DIR)/docker-compose.ollama.yml

OBS_DIR := deployments/observability
COMPOSE_OBS_LF_EXPORT := -f $(OBS_DIR)/docker-compose.observability.langfuse-export.yml
COMPOSE_LANGFUSE := -f $(OBS_DIR)/docker-compose.langfuse.yml

.PHONY: docker-up docker-down docker-up-ui docker-down-ui docker-up-vllm docker-down-vllm docker-up-ollama docker-down-ollama obs-up-llm obs-down-llm obs-up-all obs-down-all test test-integration eval-prepare eval-run eval-gate docker-build-api docker-build-worker

# Base: app infra + OTEL collect (Collector, Prometheus, Tempo, JMX, cAdvisor).
# UIs (Kibana, Kafka UI, Grafana) live in docker-compose.ui.yml.
docker-up:
	docker compose $(COMPOSE_BASE) up -d

docker-up-ui:
	docker compose $(COMPOSE_BASE) $(COMPOSE_UI) up -d

docker-down-ui:
	docker compose $(COMPOSE_BASE) $(COMPOSE_UI) stop kibana kafka-ui grafana
	docker compose $(COMPOSE_BASE) $(COMPOSE_UI) rm -f kibana kafka-ui grafana

# Two images, not one: the API never parses documents (it only enqueues a
# Kafka event on upload-complete), only the worker does - see Dockerfile.api
# vs Dockerfile.worker for the dependency split this buys.
docker-build-api:
	docker build -f Dockerfile.api -t rag-sys-api:dev .

docker-build-worker:
	docker build -f Dockerfile.worker -t rag-sys-worker:dev .

docker-down:
	docker compose $(COMPOSE_BASE) $(COMPOSE_UI) down

docker-up-vllm:
	docker compose $(COMPOSE_VLLM) up -d

docker-down-vllm:
	docker compose $(COMPOSE_VLLM) down

docker-up-ollama:
	docker compose $(COMPOSE_OLLAMA) up -d

docker-down-ollama:
	docker compose $(COMPOSE_OLLAMA) down

# Langfuse stack + fan OTEL traces into Langfuse (base collect already running).
obs-up-llm:
	docker compose $(COMPOSE_BASE) $(COMPOSE_OBS_LF_EXPORT) $(COMPOSE_LANGFUSE) --profile llm up -d

obs-down-llm:
	docker compose $(COMPOSE_BASE) $(COMPOSE_OBS_LF_EXPORT) $(COMPOSE_LANGFUSE) --profile llm stop \
		langfuse langfuse-worker langfuse-postgres langfuse-clickhouse langfuse-redis langfuse-minio
	docker compose $(COMPOSE_BASE) $(COMPOSE_OBS_LF_EXPORT) $(COMPOSE_LANGFUSE) --profile llm rm -f \
		langfuse langfuse-worker langfuse-postgres langfuse-clickhouse langfuse-redis langfuse-minio
	docker compose $(COMPOSE_BASE) up -d otel-collector

obs-up-all:
	docker compose $(COMPOSE_BASE) $(COMPOSE_UI) $(COMPOSE_OBS_LF_EXPORT) $(COMPOSE_LANGFUSE) --profile llm up -d

obs-down-all:
	docker compose $(COMPOSE_BASE) $(COMPOSE_UI) $(COMPOSE_OBS_LF_EXPORT) $(COMPOSE_LANGFUSE) --profile llm down

# --group evaluation: tests/evaluation/ imports pyarrow/aiohttp/pyyaml directly.
test:
	uv run --group evaluation poe test

test-integration:
	uv run --group evaluation poe test-integration

# --no-sync on the outer uv run too: eval-prepare/eval-run are normally used while
# `uv run app`/`uv run worker` are already running, and uv's own "does the project
# need rebuilding" check would otherwise try to overwrite app.exe/worker.exe while
# Windows still has them open as a running process.
eval-prepare:
	uv run --no-sync poe eval-prepare

eval-run:
	uv run --no-sync poe eval-run

eval-gate:
	uv run --no-sync poe eval-gate
