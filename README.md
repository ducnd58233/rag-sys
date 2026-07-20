# RAG System

## Contents

- [Setup](#setup)
- [Running](#running)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [API documentation](#api-documentation)
- [Observability and dashboards](#observability-and-dashboards)
- [File size / SHA-256 for the upload API](#file-size--sha-256-for-the-upload-api)

## Setup

```bash
conda env create -f environment.yaml   # first time only
conda activate rag-sys
```

## Running

```bash
make docker-up
make docker-up-ollama
uv run app       # or: uv run dev (auto-reload)
uv run worker
```

## Testing

```bash
make test              # fast unit tests, mocked/faked infra, no Docker required
make test-integration  # real Postgres/Elasticsearch/Neo4j via Testcontainers, needs Docker
```

## Evaluation

Retrieval quality is measured with a golden question set against a real, downloadable arXiv
corpus, scored with deterministic metrics (recall/precision/hit-rate/MRR/nDCG) - see
[`datasets/README.md`](datasets/README.md) for the corpus and dataset, and
[`scripts/evaluation/`](scripts/evaluation) for the harness itself.

```bash
uv run poe eval-download-corpus  # fetch the fixture corpus
uv run poe eval-index-corpus     # index it into a running local stack
uv run poe eval-dev              # score retrieval against the dev split
```

## API documentation

After starting the API with `uv run app` or `uv run dev`, open:

- Swagger UI (test endpoints): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

## Observability and dashboards

Full dashboard list, key Grafana panels, and the trace-span reference live in
[`deployments/observability/README.md`](deployments/observability/README.md). Quick links once
`make docker-up` (or `make obs-up`) is running:

- Grafana: http://localhost:3001
- Langfuse LLM analytics: http://localhost:3002
- GraphDB (Neo4j Browser): http://localhost:7474 (`neo4j` / `rag-sys-dev`)
- Kafka UI (Kafbat): http://localhost:8082

## File size / SHA-256 for the upload API

Bash:

```bash
stat -c%s ./file.pdf
sha256sum ./file.pdf | cut -d' ' -f1
```

PowerShell:

```powershell
(Get-Item .\file.pdf).Length
(Get-FileHash .\file.pdf -Algorithm SHA256).Hash.ToLower()
```
