# RAG System

## Contents

- [Setup](#setup)
- [Running](#running)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Retrieval strategies](#retrieval-strategies)
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

```bash
uv sync --group evaluation  # one-time per clone
make docker-up
make eval-prepare  # download -> process -> ingest -> datasets/manifest.json
make eval-run       # score retrieval against the dev split
```

See [`datasets/README.md`](datasets/README.md) for the dataset, the manifest, and the id mapping,
and [`scripts/evaluation/`](scripts/evaluation) for the harness itself.

## Retrieval strategies

The query router (`src/modules/retrieval/app/router/`) picks one or more of these per query;
see `src/modules/retrieval/infra/strategies/`. `lexical` (BM25) and `semantic` (dense vector)
are not independently selectable - `hybrid` always runs both in parallel and
reciprocal-rank-fuses the results, since the two catch different failure modes on the same
query rather than serving different query types.

- **hybrid** - fused keyword (BM25) + semantic vector search. The default for ordinary
  content questions.
- **structured** - exact-match lookup by identifier fields (`chunk_id`, `document_id`,
  `document_version_id`, `metadata.filename`, `metadata.source`), or scoped content search
  when an explicit `document_id`/`document_version_id` filter is given.
- **temporal** - lexical search boosted by recency, decaying by `valid_from` age.
- **graph** - traverses entity relationships in Neo4j to find related chunks/documents, then
  scores Elasticsearch hits by hop distance.

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
