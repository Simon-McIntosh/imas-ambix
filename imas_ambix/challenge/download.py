"""Revision-pinned, resumable downloader for the Sophelio corpus.

Files are streamed directly to their final GPFS paths.  No temporary publish
file, rename, or no-clobber rename is used; an interrupted final file is
continued with an HTTP range request on the next invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

REPOSITORY = "Sophelio/fusion-equilibrium-challenge"
REVISION = "1e280905b85f2a6fdde7e06fca8cf3a1edf447cb"
DEFAULT_ROOT = Path("/work/projects/imas_gpu/sophelio/raw")
DATA_PREFIXES = (
    "data/diii_d_train",
    "data/diii_d_public_test",
    "data/mast_public_test",
)
_API_ROOT = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
_FILE_ROOT = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
_NEXT_LINK = re.compile(r'<([^>]+)>; rel="next"')


@dataclass(frozen=True)
class InventoryItem:
    """One immutable remote corpus object."""

    path: str
    size: int
    sha256: str


def _request(url: str, *, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "imas-ambix-challenge-ingest/1", **(headers or {})},
    )
    return urllib.request.urlopen(request, timeout=120)


def iter_inventory(
    prefixes: tuple[str, ...] = DATA_PREFIXES,
) -> Iterator[InventoryItem]:
    """Yield the pinned repository inventory using the paginated tree API."""

    for prefix in prefixes:
        encoded = urllib.parse.quote(prefix, safe="")
        url: str | None = (
            f"{_API_ROOT}/{encoded}?recursive=false&expand=false&limit=1000"
        )
        while url:
            with _request(url) as response:
                page = json.load(response)
                link = response.headers.get("Link", "")
            for entry in page:
                if entry.get("type") != "file" or not entry["path"].endswith(
                    ".parquet"
                ):
                    continue
                lfs = entry.get("lfs") or {}
                digest = lfs.get("oid")
                if not digest:
                    raise ValueError(f"missing LFS digest for {entry['path']}")
                yield InventoryItem(
                    path=entry["path"], size=int(entry["size"]), sha256=digest
                )
            match = _NEXT_LINK.search(link)
            url = match.group(1) if match else None


def write_inventory(path: Path, items: list[InventoryItem]) -> None:
    """Write a complete inventory directly, with durable line boundaries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        header = {"kind": "inventory", "repository": REPOSITORY, "revision": REVISION}
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        for item in items:
            stream.write(json.dumps(asdict(item), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_inventory(path: Path) -> list[InventoryItem]:
    """Read an inventory and reject a revision mismatch."""

    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("revision") != REVISION:
            raise ValueError(
                f"inventory revision {header.get('revision')} does not match {REVISION}"
            )
        return [InventoryItem(**json.loads(line)) for line in stream if line.strip()]


class EventManifest:
    """Append-only download receipt safe for concurrent worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def completed(self) -> set[tuple[str, str, int]]:
        if not self.path.exists():
            return set()
        result: set[tuple[str, str, int]] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "complete":
                    result.add((event["path"], event["sha256"], int(event["size"])))
        return result

    def append(self, event: str, item: InventoryItem, **fields: Any) -> None:
        record = {
            "event": event,
            "path": item.path,
            "revision": REVISION,
            "sha256": item.sha256,
            "size": item.size,
            "time": time.time(),
            **fields,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_item(
    item: InventoryItem,
    root: Path,
    manifest: EventManifest,
    completed: set[tuple[str, str, int]],
) -> str:
    """Download or resume one file directly at its final path."""

    destination = root / item.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = (item.path, item.sha256, item.size)
    if (
        destination.exists()
        and destination.stat().st_size == item.size
        and (receipt in completed or _sha256(destination) == item.sha256)
    ):
        if receipt not in completed:
            manifest.append("complete", item, disposition="verified-existing")
        return "skipped"

    for attempt in range(1, 6):
        offset = destination.stat().st_size if destination.exists() else 0
        if offset > item.size:
            destination.open("wb").close()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        url = f"{_FILE_ROOT}/{urllib.parse.quote(item.path)}?download=true"
        try:
            with _request(url, headers=headers) as response:
                resumed = offset > 0 and response.status == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    offset = 0
                with destination.open(mode) as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            actual_size = destination.stat().st_size
            if actual_size != item.size:
                raise OSError(f"size {actual_size} != expected {item.size}")
            actual_digest = _sha256(destination)
            if actual_digest != item.sha256:
                destination.open("wb").close()
                raise OSError(f"sha256 {actual_digest} != expected {item.sha256}")
            manifest.append(
                "complete", item, disposition="resumed" if resumed else "downloaded"
            )
            return "downloaded"
        except (OSError, urllib.error.URLError) as exc:
            manifest.append("retry", item, attempt=attempt, error=str(exc))
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def download_corpus(root: Path, workers: int) -> dict[str, int]:
    """Materialize all released configs and return completion counts."""

    inventory_path = root / "inventory.jsonl"
    if inventory_path.exists():
        items = read_inventory(inventory_path)
    else:
        items = list(iter_inventory())
        write_inventory(inventory_path, items)
    manifest = EventManifest(root / "download-manifest.jsonl")
    completed = manifest.completed()
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    manifest.append(
        "run-start",
        InventoryItem(path=".", size=sum(item.size for item in items), sha256=""),
        objects=len(items),
        workers=workers,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_item, item, root, manifest, completed): item
            for item in items
        }
        for future in as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception as exc:  # each failure is banked; remaining files continue
                counts["failed"] += 1
                manifest.append("failed", futures[future], error=str(exc))
    manifest.append(
        "run-end",
        InventoryItem(path=".", size=sum(item.size for item in items), sha256=""),
        **counts,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    counts = download_corpus(args.root, args.workers)
    print(json.dumps(counts, sort_keys=True))
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
