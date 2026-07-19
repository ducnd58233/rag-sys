COMPOSE_DIR := deployments/docker
COMPOSE_BASE := -f $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_VLLM := -f $(COMPOSE_DIR)/docker-compose.vllm.yml
COMPOSE_OLLAMA := -f $(COMPOSE_DIR)/docker-compose.ollama.yml

OBS_DIR := deployments/observability
COMPOSE_OBS := -f $(OBS_DIR)/docker-compose.observability.yml
COMPOSE_LANGFUSE := -f $(OBS_DIR)/docker-compose.langfuse.yml

.PHONY: docker-up docker-down docker-up-vllm docker-down-vllm docker-up-ollama docker-down-ollama obs-up obs-down obs-up-llm obs-down-llm

docker-up:
	docker compose $(COMPOSE_BASE) $(COMPOSE_OBS) up -d

docker-down:
	docker compose $(COMPOSE_BASE) $(COMPOSE_OBS) down

docker-up-vllm:
	docker compose $(COMPOSE_VLLM) up -d

docker-down-vllm:
	docker compose $(COMPOSE_VLLM) down

docker-up-ollama:
	docker compose $(COMPOSE_OLLAMA) up -d

docker-down-ollama:
	docker compose $(COMPOSE_OLLAMA) down

obs-up:
	docker compose $(COMPOSE_OBS) up -d

obs-down:
	docker compose $(COMPOSE_OBS) down

obs-up-llm:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm up -d

obs-down-llm:
	docker compose $(COMPOSE_OBS) $(COMPOSE_LANGFUSE) --profile llm down
