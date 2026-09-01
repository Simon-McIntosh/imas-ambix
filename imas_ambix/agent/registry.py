"""Shared registration records for running model serves.

A registration is a route candidate, not availability evidence. Callers must
still reconcile job liveness and successfully probe the registered endpoint
before using it as an upstream.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class ServeRegistration:
    """Resolved identity written by one serving allocation."""

    model_id: str
    host: str
    port: int
    job_id: str
    accelerator_count: int
    checkpoint_precision: str

    def __post_init__(self) -> None:
        for name in ("model_id", "host", "job_id", "checkpoint_precision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not _SAFE_JOB_ID.fullmatch(self.job_id):
            raise ValueError("job_id contains characters unsafe for a record name")
        if not isinstance(self.port, int) or isinstance(self.port, bool):
            raise ValueError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if (
            not isinstance(self.accelerator_count, int)
            or isinstance(self.accelerator_count, bool)
            or self.accelerator_count < 1
        ):
            raise ValueError("accelerator_count must be a positive integer")

    @property
    def origin(self) -> str:
        """Direct HTTP origin reported by this allocation."""
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class RegistrationRead:
    """Records reconciled against job liveness, but not endpoint readiness."""

    current: tuple[ServeRegistration, ...]
    stale: tuple[ServeRegistration, ...]


def registration_directory(base_dir: str | Path) -> Path:
    """Return the shared per-serve registry directory for a site base."""
    return Path(base_dir) / "agents" / "serve-registry"


def _record_path(directory: Path, job_id: str) -> Path:
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("job_id contains characters unsafe for a record name")
    return directory / f"{job_id}.json"


def write_registration(registration: ServeRegistration, directory: str | Path) -> Path:
    """Atomically replace one serve registration and return its final path."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _record_path(target_dir, registration.job_id)
    payload = json.dumps(asdict(registration), sort_keys=True, separators=(",", ":"))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=f".{registration.job_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        directory_fd = os.open(target_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def remove_registration(job_id: str, directory: str | Path) -> None:
    """Remove one registration; repeated cleanup is harmless."""
    _record_path(Path(directory), job_id).unlink(missing_ok=True)


def read_registrations(
    directory: str | Path,
    *,
    job_is_running: Callable[[str], bool],
) -> RegistrationRead:
    """Read valid records and split them by independently checked job liveness.

    Entries in ``current`` are still only candidates. A caller must probe each
    origin before treating it as an upstream. Malformed files are ignored so a
    damaged record cannot become a route.
    """
    target_dir = Path(directory)
    current: list[ServeRegistration] = []
    stale: list[ServeRegistration] = []
    if not target_dir.is_dir():
        return RegistrationRead((), ())
    for path in sorted(target_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            registration = ServeRegistration(**raw)
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
        destination = current if job_is_running(registration.job_id) else stale
        destination.append(registration)

    def sort_key(record: ServeRegistration) -> tuple[str, int, str]:
        return record.model_id, record.accelerator_count, record.job_id

    return RegistrationRead(
        tuple(sorted(current, key=sort_key)), tuple(sorted(stale, key=sort_key))
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage one serve registration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--directory", required=True)
    write.add_argument("--model-id", required=True)
    write.add_argument("--host", required=True)
    write.add_argument("--port", required=True, type=int)
    write.add_argument("--job-id", required=True)
    write.add_argument("--accelerator-count", required=True, type=int)
    write.add_argument("--checkpoint-precision", required=True)
    remove = subparsers.add_parser("remove")
    remove.add_argument("--directory", required=True)
    remove.add_argument("--job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generated-job registration helper."""
    args = _parser().parse_args(argv)
    if args.command == "write":
        write_registration(
            ServeRegistration(
                model_id=args.model_id,
                host=args.host,
                port=args.port,
                job_id=args.job_id,
                accelerator_count=args.accelerator_count,
                checkpoint_precision=args.checkpoint_precision,
            ),
            args.directory,
        )
    else:
        remove_registration(args.job_id, args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
