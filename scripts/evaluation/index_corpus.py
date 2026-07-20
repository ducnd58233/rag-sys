"""Prime a local stack with the evaluation fixture corpus.

Uploads and ingests every file under datasets/{pdf,md,docx}/ through the app's own
composition root (src.bootstrap.build_container), calling the same IngestDocumentUseCase
the production worker eventually runs - but in-process, bypassing Kafka. There is no HTTP
endpoint to poll for "is this document indexed yet" (see docs/rag-evaluation/PLAN.md
ADR-EV-002-REV), so priming a local evaluation stack calls the use case directly instead
of waiting on the async outbox/worker pipeline. scripts/evaluation/run.py, which measures
retrieval quality, still talks to the running app over HTTP (ADR-EV-001) - only fixture
setup takes this shortcut.

Requires `make docker-up` and datasets/pdf/ already populated
(`uv run poe eval-download-corpus`).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

import aiohttp
from scripts.evaluation.config import DEFAULT_EVAL_ORG_ID, DEFAULT_EVAL_USER_ID

from src.bootstrap import AppContainer, build_container
from src.modules.document.app.dto import CompleteUploadRequest, CreateUploadUrlRequest
from src.modules.ingestion.app.dto import IngestDocumentRequest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIRS = (
    _REPO_ROOT / "datasets" / "pdf",
    _REPO_ROOT / "datasets" / "docx",
    _REPO_ROOT / "datasets" / "md",
)

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
}


def _corpus_files() -> list[Path]:
    files: list[Path] = []
    for source_dir in _SOURCE_DIRS:
        if not source_dir.exists():
            continue
        files.extend(sorted(p for p in source_dir.iterdir() if p.is_file()))
    return files


async def _index_file(
    container: AppContainer,
    session: aiohttp.ClientSession,
    path: Path,
    *,
    org_id: int,
    user_id: int,
) -> None:
    mime_type = _MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        print(f"skip  {path.name}  (unsupported extension {path.suffix})")
        return

    content = path.read_bytes()
    upload = await container.create_document_upload.execute(
        CreateUploadUrlRequest(
            org_id=org_id,
            user_id=user_id,
            filename=path.name,
            mime_type=mime_type,
            size_bytes=len(content),
        ),
    )

    async with session.put(
        upload.upload_url,
        data=content,
        headers={"Content-Type": mime_type},
    ) as response:
        response.raise_for_status()

    completed = await container.complete_document_upload.execute(
        CompleteUploadRequest(
            org_id=org_id,
            document_version_id=upload.document_version_id,
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        ),
    )

    result = await container.ingest_document.execute(
        IngestDocumentRequest(
            org_id=org_id,
            document_version_id=completed.document_version_id,
        ),
    )
    print(
        f"ok    {path.name}  document_id={result.document_id} "
        f"chunks={result.chunk_count}",
    )


async def index_corpus(*, org_id: int, user_id: int) -> int:
    files = _corpus_files()
    if not files:
        print("no corpus files found; run `uv run poe eval-download-corpus` first")
        return 0

    container = build_container(service_name="evaluation-index-corpus")
    await container.startup()
    try:
        async with aiohttp.ClientSession() as session:
            for path in files:
                await _index_file(
                    container, session, path, org_id=org_id, user_id=user_id
                )
    finally:
        await container.shutdown()

    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", type=int, default=DEFAULT_EVAL_ORG_ID)
    parser.add_argument("--user-id", type=int, default=DEFAULT_EVAL_USER_ID)
    args = parser.parse_args(argv)

    indexed = asyncio.run(index_corpus(org_id=args.org_id, user_id=args.user_id))
    if indexed:
        print(f"\nindexed {indexed} files for org_id={args.org_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
