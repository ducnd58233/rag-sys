from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "datasets" / "manifest.json"
_PDF_DIR = _REPO_ROOT / "datasets" / "pdf"
_USER_AGENT = "rag-sys-evaluation-corpus-downloader/1.0"
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 30.0


def load_manifest(manifest_path: Path = _MANIFEST_PATH) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["papers"]


def download_paper(
    paper: dict[str, str],
    *,
    destination_dir: Path,
    force: bool,
) -> Path:
    destination = destination_dir / paper["filename"]
    if destination.exists() and not force:
        return destination

    request = urllib.request.Request(
        paper["pdf_url"],
        headers={"User-Agent": _USER_AGENT},
    )
    last_error: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as response:
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError(
                    f"{paper['pdf_url']} did not return a PDF (got "
                    f"{len(payload)} bytes not starting with %PDF)",
                )
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return destination
        except (urllib.error.URLError, ValueError) as error:
            last_error = error
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Could not download {paper['title']} ({paper['arxiv_id']}) after "
        f"{_RETRY_ATTEMPTS} attempts",
    ) from last_error


def download_corpus(
    *,
    manifest_path: Path = _MANIFEST_PATH,
    destination_dir: Path = _PDF_DIR,
    force: bool = False,
) -> list[Path]:
    papers = load_manifest(manifest_path)
    downloaded: list[Path] = []
    for paper in papers:
        path = download_paper(paper, destination_dir=destination_dir, force=force)
        downloaded.append(path)
        print(f"ok  {paper['arxiv_id']}  {path.name}")
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

    print(f"\n{len(downloaded)} papers available in {_PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
