# RAG System

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

## API documentation

After starting the API with `uv run app` or `uv run dev`, open:

- Swagger UI (test endpoints): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

## Complex RAG answering

`/generation/ask` uses bounded query decomposition for compound questions when
`CHAT_COMPLEX_RAG_ENABLED=true`. Simple questions keep the single-hop path.

Key local settings:

```bash
CHAT_TOP_K=8
CHAT_COMPLEX_RAG_ENABLED=true
CHAT_QUERY_DECOMPOSITION_MAX_SUBQUESTIONS=4
CHAT_COMPLEX_RAG_MAX_RETRIEVAL_QUERIES=5
CHAT_COMPLEX_RAG_PER_QUERY_TOP_K=6
CHAT_COMPLEX_RAG_FINAL_TOP_K=12
CHAT_COMPLEX_RAG_MAX_ITERATIONS=1
```

Manual smoke after ingesting Transformer source material:

```powershell
$body = @{
  org_id = 1
  query = "What is transformers and how to calculate attention?"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/generation/ask `
  -ContentType "application/json" `
  -Body $body
```

The answer should cover both Transformer architecture and scaled dot-product
attention calculation, with citations for both supported parts.

## Dashboards

- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090
- Tempo: http://localhost:3200
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
