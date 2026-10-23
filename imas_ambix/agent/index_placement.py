"""Node-local watch index placement and verified durable snapshots."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from compression import zstd

if TYPE_CHECKING:
    from imas_ambix.agent.telemetry_index import TelemetryIndex

NETWORK_FILESYSTEM_TYPES = frozenset(
    {
        0x47504653,  # GPFS
        0x6969,  # NFS
        0xFE534D42,  # SMB
        0x0BD00BD0,  # Lustre
    }
)
SNAPSHOT_DIR = Path.home() / ".local" / "state" / "ambix" / "index-snapshots"
SNAPSHOT_PREFIX = "watch-index-"
SNAPSHOT_SUFFIX = ".sqlite3.zst"
_VERIFIED_SIGNATURES: dict[Path, tuple[int, int, int, int, int]] = {}


def filesystem_type(path: str | Path) -> int:
    """Return Linux statfs f_type for the nearest existing parent of *path*."""
    ancestor = Path(path).expanduser().resolve()
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise FileNotFoundError(path)
        ancestor = ancestor.parent
    # f_type is the first native long of struct statfs on Linux.
    buffer = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.statfs(os.fsencode(ancestor), buffer) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(ancestor))
    return ctypes.c_ulong.from_buffer_copy(buffer).value


def require_local(path: str | Path) -> Path:
    """Refuse a network filesystem before SQLite can enable WAL there."""
    target = Path(path).expanduser()
    kind = filesystem_type(target)
    if kind in NETWORK_FILESYSTEM_TYPES:
        raise ValueError(
            f"watch index requires local disk for WAL: {target} (statfs={kind:#x})"
        )
    return target


def resolve_index_path(explicit: str | Path | None = None) -> Path | None:
    """Resolve explicit path, environment override, then fleet local runtime."""
    if explicit:
        return require_local(explicit)
    override = os.environ.get("AMBIX_WATCH_INDEX")
    if override:
        return require_local(override)
    runtime = os.environ.get("FLEET_RUNTIME_DIR")
    if runtime:
        root = Path(runtime).expanduser()
        candidate = root / "ambix" / "watch-index.sqlite3"
        if root.is_dir() and filesystem_type(candidate) not in NETWORK_FILESYSTEM_TYPES:
            return candidate
    return None


def enable_wal(index: TelemetryIndex) -> None:
    """Set WAL on a local writer and verify SQLite accepted it."""
    require_local(index.path)
    mode = index._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if mode.lower() != "wal":
        raise RuntimeError(f"SQLite refused WAL at {index.path}: {mode}")


def _snapshot_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"{SNAPSHOT_PREFIX}*{SNAPSHOT_SUFFIX}"), reverse=True)


def _decompress_and_check(source: Path, destination: Path) -> None:
    with zstd.open(source, "rb") as compressed, destination.open("wb") as plain:
        shutil.copyfileobj(compressed, plain, length=1024 * 1024)
    with sqlite3.connect(f"file:{destination}?mode=ro", uri=True) as checked:
        result = checked.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise ValueError(f"snapshot quick_check failed: {source}: {result}")


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _snapshot_is_verified(source: Path, local_dir: Path) -> bool:
    """Verify changed snapshot bytes, reusing only an unchanged prior verdict."""
    try:
        signature = _signature(source)
        if _VERIFIED_SIGNATURES.get(source) == signature:
            return True
        with tempfile.TemporaryDirectory(dir=local_dir) as scratch:
            _decompress_and_check(source, Path(scratch) / "check.sqlite3")
        if _signature(source) != signature:
            raise ValueError("snapshot changed while it was checked")
    except (OSError, ValueError, sqlite3.DatabaseError, zstd.ZstdError) as error:
        _VERIFIED_SIGNATURES.pop(source, None)
        print(
            f"snapshot verification failed: {source}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return False
    _VERIFIED_SIGNATURES[source] = signature
    return True


def _verified_snapshots(directory: Path, local_dir: Path) -> list[Path]:
    verified = []
    for source in _snapshot_paths(directory):
        if _snapshot_is_verified(source, local_dir):
            verified.append(source)
    return verified


def snapshot_index(
    index: TelemetryIndex | Path,
    *,
    directory: Path = SNAPSHOT_DIR,
    now: datetime | None = None,
) -> Path:
    """Back up online, compress, verify, then retain three verified copies."""
    index_path = require_local(index.path if hasattr(index, "path") else index)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    final = directory / f"{SNAPSHOT_PREFIX}{stamp}{SNAPSHOT_SUFFIX}"
    with tempfile.TemporaryDirectory(dir=index_path.parent) as scratch:
        plain = Path(scratch) / "backup.sqlite3"
        with (
            closing(
                sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
            ) as source,
            closing(sqlite3.connect(plain)) as destination,
        ):
            destination.execute("PRAGMA journal_mode=OFF")
            destination.execute("PRAGMA synchronous=OFF")
            source.backup(destination)
            destination.execute("PRAGMA journal_mode=OFF")
            destination.execute("PRAGMA synchronous=OFF")
            result = destination.execute("PRAGMA quick_check").fetchone()
            if result is None or result[0] != "ok":
                raise ValueError(f"snapshot scratch quick_check failed: {result}")
        staged = directory / f".{final.name}.{os.getpid()}.tmp"
        try:
            digest = hashlib.sha256()
            with plain.open("rb") as source, zstd.open(staged, "wb") as compressed:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    compressed.write(chunk)
            with staged.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(staged, final)
            signature = _signature(final)
            verified = hashlib.sha256()
            with zstd.open(final, "rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    verified.update(chunk)
            if verified.digest() != digest.digest():
                raise ValueError(f"snapshot digest mismatch: {final}")
            if _signature(final) != signature:
                raise ValueError("snapshot changed while it was checked")
            _VERIFIED_SIGNATURES[final] = signature
        except Exception:
            final.unlink(missing_ok=True)
            raise
        finally:
            staged.unlink(missing_ok=True)
        verified = _verified_snapshots(directory, index_path.parent)
        for old in verified[3:]:
            old.unlink()
    return final


def restore_index(path: Path, *, directory: Path = SNAPSHOT_DIR) -> Path | None:
    """Restore the newest valid copy, falling back on damage."""
    target = require_local(path)
    if target.exists() or not directory.is_dir():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    for source in _snapshot_paths(directory):
        with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
            candidate = Path(scratch) / "restored.sqlite3"
            try:
                _decompress_and_check(source, candidate)
                os.replace(candidate, target)
            except (
                OSError,
                ValueError,
                sqlite3.DatabaseError,
                zstd.ZstdError,
            ) as error:
                print(
                    f"snapshot restore failed: {source}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                return source
    return None


def snapshot_if_due(
    index: TelemetryIndex | Path, *, directory: Path = SNAPSHOT_DIR
) -> Path | None:
    """Take one snapshot per UTC day after an ingest tick has committed."""
    index_path = Path(index.path if hasattr(index, "path") else index)
    with closing(
        sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as source:
        has_samples = source.execute("SELECT 1 FROM sample LIMIT 1").fetchone()
    if has_samples is None:
        return None
    today = datetime.now(UTC).strftime("%Y%m%d")
    for path in _snapshot_paths(directory):
        if path.name.startswith(f"{SNAPSHOT_PREFIX}{today}T") and _snapshot_is_verified(
            path, index_path.parent
        ):
            return None
    return snapshot_index(index_path, directory=directory)


class SnapshotScheduler:
    """Launch at most one snapshot worker while ticks continue on the writer."""

    def __init__(self, index_path: Path, *, directory: Path = SNAPSHOT_DIR) -> None:
        self.index_path = index_path
        self.directory = directory
        self._thread: threading.Thread | None = None

    def schedule(self) -> bool:
        """Queue a snapshot when no prior worker is still running."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._thread = threading.Thread(target=self._run, name="watch-index-snapshot")
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            snapshot_if_due(self.index_path, directory=self.directory)
        except Exception as error:
            print(f"snapshot failed: {error}", file=sys.stderr, flush=True)

    def join(self) -> None:
        """Let an in-flight snapshot finish before the ingest process exits."""
        if self._thread is not None:
            self._thread.join()
