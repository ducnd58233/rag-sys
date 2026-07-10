COMPOSE_DIR := deployments/docker
COMPOSE_BASE := -f $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_VLLM := -f $(COMPOSE_DIR)/docker-compose.vllm.yml
COMPOSE_OLLAMA := -f $(COMPOSE_DIR)/docker-compose.ollama.yml

.PHONY: docker-up docker-down docker-up-vllm docker-down-vllm docker-up-ollama docker-down-ollama

docker-up:
	docker compose $(COMPOSE_BASE) up -d

docker-down:
	docker compose $(COMPOSE_BASE) down

docker-up-vllm:
	docker compose $(COMPOSE_VLLM) up -d

docker-down-vllm:
	docker compose $(COMPOSE_VLLM) down

docker-up-ollama:
	docker compose $(COMPOSE_OLLAMA) up -d

docker-down-ollama:
	docker compose $(COMPOSE_OLLAMA) down