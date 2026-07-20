from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _REPO_ROOT / "datasets" / "bioasq" / "raw"
_USER_AGENT = "rag-sys-evaluation-corpus-downloader/1.0"
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 120.0
_PARQUET_MAGIC = b"PAR1"

_HF_API_ROOT = (
    "https://huggingface.co/api/datasets/rag-datasets/rag-mini-bioasq/parquet"
)
_FILES = {
    "question-answer-passages.parquet": f"{_HF_API_ROOT}/question-answer-passages/test/0.parquet",
    "text-corpus.parquet": f"{_HF_API_ROOT}/text-corpus/passages/0.parquet",
}


def download_file(url: str, *, destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                payload = response.read()
            if not payload.startswith(_PARQUET_MAGIC):
                raise ValueError(
                    f"{url} did not return a parquet file (missing PAR1 magic header)",
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return destination
        except (urllib.error.URLError, ValueError) as error:
            last_error = error
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Could not download {url} after {_RETRY_ATTEMPTS} attempts",
    ) from last_error


def download_corpus(
    *,
    destination_dir: Path = _RAW_DIR,
    force: bool = False,
) -> list[Path]:
    downloaded: list[Path] = []
    for filename, url in _FILES.items():
        path = download_file(url, destination=destination_dir / filename, force=force)
        downloaded.append(path)
        print(f"ok  {filename}  ({path.stat().st_size:,} bytes)")
    return downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the file already exists",
    )
    args = parser.parse_args(argv)

    try:
        downloaded = download_corpus(force=args.force)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"\n{len(downloaded)} files available in {_RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
