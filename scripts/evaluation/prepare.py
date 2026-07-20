from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from scripts.evaluation.config import DEFAULT_EVAL_ORG_ID, DEFAULT_EVAL_USER_ID
from scripts.evaluation.manifest import DatasetManifest, dataset_dir, progress_path
from scripts.evaluation.manifest import save as save_manifest
from scripts.evaluation.sources import SOURCES, get_source
from tqdm import tqdm

from src.bootstrap import AppContainer, build_container
from src.modules.document.app.dto import CompleteUploadRequest, CreateUploadUrlRequest
from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import LoggingSettings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = "bioasq"
_DEFAULT_CONCURRENCY = 10


def _load_progress(progress_file: Path) -> dict[str, str]:
    if not progress_file.exists():
        return {}
    id_map: dict[str, str] = {}
    for line in progress_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        id_map[entry["passage_id"]] = entry["document_id"]
    return id_map


async def _ingest_one(
    container: AppContainer,
    session: aiohttp.ClientSession,
    *,
    org_id: int,
    user_id: int,
    filename: str,
    content: bytes,
    mime_type: str,
) -> int:
    """Uploads and ingests in-process via the app's own use cases (no HTTP/Kafka round trip); returns the DB document_id."""
    upload = await container.create_document_upload.execute(
        CreateUploadUrlRequest(
            org_id=org_id,
            user_id=user_id,
            filename=filename,
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
    return result.document_id


async def prepare(
    *,
    dataset: str,
    org_id: int,
    user_id: int,
    corpus_limit: int | None,
    corpus_fraction: float | None,
    concurrency: int,
    force_download: bool,
    resume: bool,
) -> Path:
    source = get_source(dataset)
    raw_dir = dataset_dir(dataset) / "raw"
    processed_dir = dataset_dir(dataset) / "processed"
    progress_file = progress_path(dataset)

    logger.info("[1/3] downloading %s into %s", dataset, raw_dir)
    source.download(raw_dir, force=force_download)

    passages = source.iter_corpus_passages(
        raw_dir,
        limit=corpus_limit,
        fraction=corpus_fraction,
    )
    logger.info("[2/3] %d passages selected for indexing", len(passages))

    id_map: dict[str, str] = _load_progress(progress_file) if resume else {}
    pending = [passage for passage in passages if passage.id not in id_map]
    logger.info(
        "%d already indexed (resumed from %s), %d remaining",
        len(passages) - len(pending),
        progress_file.name,
        len(pending),
    )

    failed: list[str] = []
    if pending:
        semaphore = asyncio.Semaphore(concurrency)
        processed_dir.mkdir(parents=True, exist_ok=True)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        container = build_container(service_name=f"evaluation-prepare-{dataset}")
        try:
            await container.startup()
            async with aiohttp.ClientSession() as session:
                with progress_file.open("a", encoding="utf-8") as progress_handle:
                    progress_bar = tqdm(
                        total=len(pending),
                        desc="ingesting",
                        unit="doc",
                    )

                    async def _run_one(passage) -> None:  # noqa: ANN001
                        async with semaphore:
                            processed_path = processed_dir / source.processed_filename(
                                passage.id
                            )
                            processed_path.write_text(
                                passage.text,
                                encoding="utf-8",
                            )
                            content = processed_path.read_bytes()

                            try:
                                document_id = await _ingest_one(
                                    container,
                                    session,
                                    org_id=org_id,
                                    user_id=user_id,
                                    filename=processed_path.name,
                                    content=content,
                                    mime_type=source.MIME_TYPE,
                                )
                            except Exception:
                                logger.exception(
                                    "failed to ingest passage %s",
                                    passage.id,
                                )
                                failed.append(passage.id)
                                progress_bar.update(1)
                                return

                            id_map[passage.id] = str(document_id)
                            progress_handle.write(
                                json.dumps(
                                    {
                                        "passage_id": passage.id,
                                        "document_id": str(document_id),
                                    },
                                )
                                + "\n",
                            )
                            progress_handle.flush()
                            progress_bar.update(1)

                    await asyncio.gather(
                        *(_run_one(passage) for passage in pending),
                    )
                    progress_bar.close()
        finally:
            await container.shutdown()

    if failed:
        logger.warning(
            "%d passages failed to ingest: %s%s",
            len(failed),
            failed[:20],
            " ..." if len(failed) > 20 else "",
        )

    logger.info("[3/3] writing manifest")
    dataset_provenance = source.provenance(raw_dir)
    manifest = DatasetManifest(
        dataset=dataset,
        created_at=datetime.now(timezone.utc).isoformat(),
        org_id=org_id,
        user_id=user_id,
        corpus_limit=corpus_limit,
        corpus_fraction=corpus_fraction,
        # as_posix() keeps manifest.json's paths parseable on every OS, not just Windows.
        raw_dir=raw_dir.relative_to(_REPO_ROOT).as_posix(),
        processed_dir=processed_dir.relative_to(_REPO_ROOT).as_posix(),
        source=str(dataset_provenance["source"]),
        source_license=str(dataset_provenance["source_license"]),
        qa_content_hash=str(dataset_provenance["qa_content_hash"]),
        corpus_content_hash=str(dataset_provenance["corpus_content_hash"]),
        passages_available=len(passages),
        passages_indexed=len(id_map),
        passages_failed=failed,
        id_map=id_map,
    )
    manifest_path = save_manifest(manifest)
    logger.info(
        "wrote %s (%d/%d passages mapped, org_id=%d)",
        manifest_path,
        len(id_map),
        len(passages),
        org_id,
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    configure_logging(
        LoggingSettings(handlers=["console"]),
        service_name="evaluation-prepare",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=_DEFAULT_DATASET, choices=sorted(SOURCES))
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
            "exclusive with --corpus-limit. A partial-corpus prepare is not comparable "
            "to a full-corpus one: retrieval difficulty depends on how many distractor "
            "passages compete with the relevant ones, and QA cases whose relevant "
            "passages fall outside the selected corpus are dropped by run.py."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download raw files even if they already exist",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "ignore the existing progress file and re-ingest every selected passage "
            "from scratch, creating new documents rather than reusing prior ones"
        ),
    )
    args = parser.parse_args(argv)

    if args.corpus_limit is not None and args.corpus_fraction is not None:
        parser.error("--corpus-limit and --corpus-fraction are mutually exclusive")

    asyncio.run(
        prepare(
            dataset=args.dataset,
            org_id=args.org_id,
            user_id=args.user_id,
            corpus_limit=args.corpus_limit,
            corpus_fraction=args.corpus_fraction,
            concurrency=args.concurrency,
            force_download=args.force_download,
            resume=not args.no_resume,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
