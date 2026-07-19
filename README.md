# RAG System

## Setup

```bash
conda env create -f environment.yaml   # first time only
conda activate rag-sys
```

## Running

1. Start infrastructure (Postgres, MinIO, Elasticsearch, Kafka):

   ```bash
   make docker-up
   make docker-up-ollama
   ```

2. Start the API:

   ```bash
   uv run app       # or: uv run dev (auto-reload)
   ```

3. Start the ingestion worker:

   ```bash
   uv run worker
   ```

The API and worker are separate processes started via `uv`, not Docker Compose services;
Compose only provisions infrastructure.

## Getting file size and SHA-256 for the upload API

The upload flow needs `size_bytes` to request an upload URL and `checksum_sha256` to complete it.

Bash:

```bash
stat -c%s ./file.pdf                # size_bytes
sha256sum ./file.pdf | cut -d' ' -f1  # checksum_sha256
```

PowerShell:

```powershell
(Get-Item .\file.pdf).Length                          # size_bytes
(Get-FileHash .\file.pdf -Algorithm SHA256).Hash.ToLower()  # checksum_sha256
```
