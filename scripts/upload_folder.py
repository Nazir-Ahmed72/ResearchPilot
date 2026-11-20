"""Upload every PDF in a folder to a running ResearchPilot API.

Browsers (and the Swagger page) cannot select a whole folder, so this sends the folder's PDFs in
batches through the normal upload endpoint (so all validation and duplicate checks still apply).

Usage:
    python scripts/upload_folder.py "C:\\Users\\me\\Downloads\\papers" --collection rag-papers
    python scripts/upload_folder.py ./papers --recursive --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx

DEFAULT_API = "http://127.0.0.1:8000"
BATCH_SIZE = 10  # the API accepts up to MAX_FILES_PER_UPLOAD (default 10) files per request


def find_pdfs(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")


def batches(items: list[Path], size: int) -> Iterator[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def upload_batch(
    client: httpx.Client, api: str, pdfs: list[Path], collection: str | None
) -> list[dict]:
    handles = [p.open("rb") for p in pdfs]
    try:
        files = [
            ("files", (p.name, h, "application/pdf")) for p, h in zip(pdfs, handles, strict=True)
        ]
        data = {"collection": collection} if collection else {}
        resp = client.post(f"{api}/api/v1/papers/upload", files=files, data=data)
    finally:
        for h in handles:
            h.close()
    body = resp.json()
    if resp.status_code == 202:
        return body["results"]
    # Nothing accepted (e.g. all duplicates): per-file results are in the error details.
    details = body.get("error", {}).get("details")
    if isinstance(details, list):
        return [
            {
                "filename": d.get("filename"),
                "status": d.get("error_code") or "rejected",
                "paper_id": d.get("paper_id"),
                "message": d.get("message"),
            }
            for d in details
        ]
    raise RuntimeError(f"Upload failed ({resp.status_code}): {body}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload all PDFs in a folder")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--collection", default=None, help="collection name (created if new)")
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    parser.add_argument("--api", default=DEFAULT_API)
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        print(f"Not a folder: {args.folder}")
        return 1
    pdfs = find_pdfs(args.folder, args.recursive)
    if not pdfs:
        print(f"No PDF files found in {args.folder}")
        return 1
    print(f"Found {len(pdfs)} PDF(s); uploading in batches of {BATCH_SIZE}...")

    counts: dict[str, int] = {}
    with httpx.Client(timeout=300) as client:
        for batch in batches(pdfs, BATCH_SIZE):
            for r in upload_batch(client, args.api, batch, args.collection):
                status = r.get("status", "?")
                counts[status] = counts.get(status, 0) + 1
                note = f" - {r['message']}" if r.get("message") else ""
                print(f"  {status:<10} {r.get('filename')}{note}")

    print("Summary:", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print("Check progress: GET /api/v1/papers (status becomes 'ready' when processed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
