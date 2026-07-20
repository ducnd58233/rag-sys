from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

import aiohttp
import pyarrow.parquet as pq
from scripts.evaluation.config import DEFAULT_EVAL_ORG_ID, DEFAULT_EVAL_USER_ID

from src.bootstrap import AppContainer, build_container
from src.modules.document.app.dto import CompleteUploadRequest, CreateUploadUrlRequest
from src.modules.ingestion.app.dto import IngestDocumentRequest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PARQUET = _REPO_ROOT / "datasets" / "bioasq" / "raw" / "text-corpus.parquet"
_PROGRESS_PATH = _REPO_ROOT / "datasets" / "bioasq" / ".index-progress.txt"
_MIME_TYPE = "text/markdown"
_DEFAULT_CONCURRENCY = 10


def _load_passages(
    parquet_path: Path,
    *,
    limit: int | None,
    fraction: float | None,
) -> list[tuple[int, str]]:
    rows = pq.read_table(parquet_path).to_pylist()
    rows.sort(key=lambda row: row["id"])
    passages = [(row["id"], row["passage"]) for row in rows]

    if fraction is not None:
        # Deterministic on passage id, not random.sample: the same fraction value always
        # selects the same passages across repeated runs, so a partial index is reproducible.
        cutoff = int(0xFFFFFFFF * fraction)
        passages = [
            (pid, text)
            for pid, text in passages
            if int(hashlib.sha256(str(pid).encode("utf-8")).hexdigest()[:8], 16)
            < cutoff
        ]
    if limit is not None:
        passages = passages[:limit]
    return passages


def _load_progress(progress_path: Path) -> set[int]:
    if not progress_path.exists():
        return set()
    return {
        int(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


async def _index_passage(
    container: AppContainer,
    session: aiohttp.ClientSession,
    passage_id: int,
    text: str,
    *,
    org_id: int,
    user_id: int,
) -> None:
    content = text.encode("utf-8")
    filename = f"bioasq-passage-{passage_id}.md"

    upload = await container.create_document_upload.execute(
        CreateUploadUrlRequest(
            org_id=org_id,
            user_id=user_id,
            filename=filename,
            mime_type=_MIME_TYPE,
            size_bytes=len(content),
        ),
    )

    async with session.put(
        upload.upload_url,
        data=content,
        headers={"Content-Type": _MIME_TYPE},
    ) as response:
        response.raise_for_status()

    completed = await container.complete_document_upload.execute(
        CompleteUploadRequest(
            org_id=org_id,
            document_version_id=upload.document_version_id,
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        ),
    )

    # Same in-process shortcut as the old corpus loader: calls the worker's own use case
    # directly instead of enqueueing and waiting on Kafka, since there is no HTTP endpoint
    # to poll for "is this document indexed yet". The eval runner (run.py) still measures
    # over HTTP, not through this path.
    await container.ingest_document.execute(
        IngestDocumentRequest(
            org_id=org_id,
            document_version_id=completed.document_version_id,
        ),
    )


async def index_corpus(
    *,
    org_id: int,
    user_id: int,
    corpus_limit: int | None,
    corpus_fraction: float | None,
    concurrency: int,
    resume: bool,
) -> int:
    if not _CORPUS_PARQUET.exists():
        print(
            f"{_CORPUS_PARQUET} not found; run `uv run poe eval-download-corpus` first"
        )
        return 0

    passages = _load_passages(
        _CORPUS_PARQUET, limit=corpus_limit, fraction=corpus_fraction
    )
    if not passages:
        print("no passages selected")
        return 0

    already_done = _load_progress(_PROGRESS_PATH) if resume else set()
    pending = [(pid, text) for pid, text in passages if pid not in already_done]
    print(
        f"{len(passages)} passages selected, {len(already_done)} already indexed, "
        f"{len(pending)} remaining",
    )

    semaphore = asyncio.Semaphore(concurrency)
    container = build_container(service_name="evaluation-index-corpus")
    await container.startup()

    indexed = 0
    failed: list[int] = []
    try:
        async with aiohttp.ClientSession() as session:
            _PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _PROGRESS_PATH.open("a", encoding="utf-8") as progress_handle:

                async def _run_one(passage_id: int, text: str) -> None:
                    nonlocal indexed
                    async with semaphore:
                        try:
                            await _index_passage(
                                container,
                                session,
                                passage_id,
                                text,
                                org_id=org_id,
                                user_id=user_id,
                            )
                        except (
                            Exception
                        ) as error:  # noqa: BLE001 - one bad passage must not abort the batch
                            failed.append(passage_id)
                            print(f"FAIL  bioasq-passage-{passage_id}  {error}")
                            return
                        progress_handle.write(f"{passage_id}\n")
                        progress_handle.flush()
                        indexed += 1
                        if indexed % 200 == 0:
                            print(f"...indexed {indexed}/{len(pending)}")

                await asyncio.gather(*(_run_one(pid, text) for pid, text in pending))
    finally:
        await container.shutdown()

    print(f"\nindexed {indexed}/{len(pending)} passages for org_id={org_id}")
    if failed:
        print(
            f"{len(failed)} passages failed: {failed[:20]}{' ...' if len(failed) > 20 else ''}"
        )
    return indexed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org-id", type=int, default=DEFAULT_EVAL_ORG_ID)
    parser.add_argument("--user-id", type=int, default=DEFAULT_EVAL_USER_ID)
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help="index only the first N passages (fast local iteration, not the fair default)",
    )
    parser.add_argument(
        "--corpus-fraction",
        type=float,
        default=None,
        help=(
            "index a deterministic fraction of the corpus, e.g. 0.1 for ~10%%. Mutually "
            "exclusive with --corpus-limit. Results from a partial-corpus index are not "
            "comparable to a full-corpus run: retrieval difficulty depends on how many "
            "distractor passages compete with the relevant ones."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore .index-progress.txt and re-index everything from scratch",
    )
    args = parser.parse_args(argv)

    if args.corpus_limit is not None and args.corpus_fraction is not None:
        parser.error("--corpus-limit and --corpus-fraction are mutually exclusive")

    asyncio.run(
        index_corpus(
            org_id=args.org_id,
            user_id=args.user_id,
            corpus_limit=args.corpus_limit,
            corpus_fraction=args.corpus_fraction,
            concurrency=args.concurrency,
            resume=not args.no_resume,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
