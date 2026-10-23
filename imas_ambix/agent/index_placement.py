"""Node-local watch index placement and verified durable snapshots."""

from __future__ import annotations

import ctypes
import os
import shutil
import sqlite3
import sys
import tempfile
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
        if root.is_dir() and filesystem_type(root) not in NETWORK_FILESYSTEM_TYPES:
            return root / "ambix" / "watch-index.sqlite3"
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


def _verified_snapshots(directory: Path, local_dir: Path) -> list[Path]:
    verified = []
    for source in _snapshot_paths(directory):
        with tempfile.TemporaryDirectory(dir=local_dir) as scratch:
            try:
                _decompress_and_check(source, Path(scratch) / "check.sqlite3")
            except (
                OSError,
                ValueError,
                sqlite3.DatabaseError,
                zstd.ZstdError,
            ) as error:
                print(
                    f"snapshot verification failed: {source}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                verified.append(source)
    return verified


def snapshot_index(
    index: TelemetryIndex,
    *,
    directory: Path = SNAPSHOT_DIR,
    now: datetime | None = None,
) -> Path:
    """Back up online, compress, verify, then retain three verified copies."""
    require_local(index.path)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    final = directory / f"{SNAPSHOT_PREFIX}{stamp}{SNAPSHOT_SUFFIX}"
    with tempfile.TemporaryDirectory(dir=index.path.parent) as scratch:
        plain = Path(scratch) / "backup.sqlite3"
        with sqlite3.connect(plain) as destination:
            index._conn.backup(destination)
        staged = directory / f".{final.name}.{os.getpid()}.tmp"
        try:
            with plain.open("rb") as source, zstd.open(staged, "wb") as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
            with staged.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(staged, final)
            check = Path(scratch) / "check.sqlite3"
            _decompress_and_check(final, check)
        finally:
            staged.unlink(missing_ok=True)
        verified = _verified_snapshots(directory, index.path.parent)
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
    index: TelemetryIndex, *, directory: Path = SNAPSHOT_DIR
) -> Path | None:
    """Take one snapshot per UTC day after an ingest tick has committed."""
    if index._conn.execute("SELECT 1 FROM sample LIMIT 1").fetchone() is None:
        return None
    today = datetime.now(UTC).strftime("%Y%m%d")
    if any(
        path.name.startswith(f"{SNAPSHOT_PREFIX}{today}T")
        for path in _snapshot_paths(directory)
    ):
        return None
    return snapshot_index(index, directory=directory)
