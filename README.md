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
