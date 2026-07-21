COMPOSE_DIR := deployments/docker
COMPOSE_BASE := -f $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_VLLM := -f $(COMPOSE_DIR)/docker-compose.vllm.yml
COMPOSE_OLLAMA := -f $(COMPOSE_DIR)/docker-compose.ollama.yml

OBS_DIR := deployments/observability
COMPOSE_OBS := -f $(OBS_DIR)/docker-compose.observability.yml
COMPOSE_LANGFUSE := -f $(OBS_DIR)/docker-compose.langfuse.yml

.PHONY: docker-up docker-down docker-up-vllm docker-down-vllm docker-up-ollama docker-down-ollama obs-up obs-down obs-up-core obs-down-core obs-up-llm obs-down-llm test test-integration eval-prepare eval-run eval-gate docker-build-api docker-build-worker

docker-up:
	docker compose $(COMPOSE_BASE) up -d
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm up -d

# Two images, not one: the API never parses documents (it only enqueues a
# Kafka event on upload-complete), only the worker does - see Dockerfile.api
# vs Dockerfile.worker for the dependency split this buys.
docker-build-api:
	docker build -f Dockerfile.api -t rag-sys-api:dev .

docker-build-worker:
	docker build -f Dockerfile.worker -t rag-sys-worker:dev .

docker-down:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm down
	docker compose $(COMPOSE_BASE) down

docker-up-vllm:
	docker compose $(COMPOSE_VLLM) up -d

docker-down-vllm:
	docker compose $(COMPOSE_VLLM) down

docker-up-ollama:
	docker compose $(COMPOSE_OLLAMA) up -d

docker-down-ollama:
	docker compose $(COMPOSE_OLLAMA) down

obs-up:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm up -d

obs-down:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm down

obs-up-core:
	docker compose $(COMPOSE_OBS) up -d

obs-down-core:
	docker compose $(COMPOSE_OBS) down

obs-up-llm:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm up -d

obs-down-llm:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm down

# --group evaluation: tests/evaluation/ imports pyarrow/aiohttp/pyyaml directly.
test:
	uv run --group evaluation poe test

test-integration:
	uv run --group evaluation poe test-integration

eval-prepare:
	uv run --no-sync poe eval-prepare

eval-run:
	uv run --no-sync poe eval-run

eval-gate:
	uv run --no-sync poe eval-gate
