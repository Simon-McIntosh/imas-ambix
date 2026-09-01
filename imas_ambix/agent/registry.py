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
from urllib.parse import urlsplit

from imas_ambix.agent.vllm_catalog import validate_catalog_metadata

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


@dataclass(frozen=True, slots=True)
class PublishedEndpoint:
    """One anonymously reachable engine exposed to standalone launchers."""

    model_id: str
    host: str
    port: int
    accelerator_family: str
    accelerator_count: int
    checkpoint_precision: str
    max_context: int

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "host",
            "accelerator_family",
            "checkpoint_precision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError(f"{name} contains control characters")
        if not isinstance(self.port, int) or isinstance(self.port, bool):
            raise ValueError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        validate_catalog_metadata(
            {
                self.model_id: {
                    "accelerator_family": self.accelerator_family,
                    "accelerator_count": self.accelerator_count,
                    "checkpoint_precision": self.checkpoint_precision,
                }
            }
        )
        if (
            not isinstance(self.max_context, int)
            or isinstance(self.max_context, bool)
            or self.max_context < 1
        ):
            raise ValueError("max_context must be a positive integer")

    @property
    def origin(self) -> str:
        """Direct HTTP origin for this engine."""
        return f"http://{self.host}:{self.port}"


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


def _endpoint_from_catalog(
    registration: ServeRegistration, payload: object
) -> PublishedEndpoint:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("endpoint catalog must contain a data list")
    matches = [
        item
        for item in payload["data"]
        if isinstance(item, dict) and item.get("id") == registration.model_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "endpoint catalog must contain the registered model exactly once"
        )
    item = matches[0]
    metadata = validate_catalog_metadata(
        {registration.model_id: item.get("ambix")}
    )[registration.model_id]
    if metadata["accelerator_count"] != registration.accelerator_count:
        raise ValueError("endpoint accelerator count disagrees with registration")
    if metadata["checkpoint_precision"] != registration.checkpoint_precision:
        raise ValueError("endpoint checkpoint precision disagrees with registration")
    parsed = urlsplit(registration.origin)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError("registration origin is incomplete")
    return PublishedEndpoint(
        model_id=registration.model_id,
        host=parsed.hostname,
        port=parsed.port,
        accelerator_family=str(metadata["accelerator_family"]),
        accelerator_count=registration.accelerator_count,
        checkpoint_precision=registration.checkpoint_precision,
        max_context=item.get("max_model_len"),
    )


def build_endpoint_document(
    registrations: Sequence[ServeRegistration],
    *,
    fetch_catalog: Callable[[str], object],
) -> tuple[PublishedEndpoint, ...]:
    """Probe registrations anonymously and retain complete live endpoints."""
    endpoints: list[PublishedEndpoint] = []
    seen_models: set[str] = set()
    for registration in registrations:
        try:
            endpoint = _endpoint_from_catalog(
                registration, fetch_catalog(registration.origin)
            )
        except (OSError, TypeError, ValueError):
            continue
        if endpoint.model_id in seen_models:
            raise ValueError(f"multiple live endpoints publish {endpoint.model_id!r}")
        endpoints.append(endpoint)
        seen_models.add(endpoint.model_id)
    return tuple(sorted(endpoints, key=lambda endpoint: endpoint.model_id))


def write_endpoint_document(
    endpoints: Sequence[PublishedEndpoint], path: str | Path
) -> Path:
    """Atomically publish launcher endpoints as a world-readable document."""
    target = Path(path)
    missing_parents: list[Path] = []
    parent = target.parent
    while not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent
    for directory in reversed(missing_parents):
        directory.mkdir(mode=0o755)
        directory.chmod(0o755)

    payload = json.dumps(
        {"endpoints": [asdict(endpoint) for endpoint in endpoints]},
        sort_keys=True,
        separators=(",", ":"),
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def read_endpoint_document(path: str | Path) -> tuple[PublishedEndpoint, ...]:
    """Read a complete endpoint document, rejecting the whole malformed file."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"endpoint document is unreadable: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("endpoints"), list):
        raise ValueError("endpoint document must contain an endpoints list")
    try:
        endpoints = tuple(PublishedEndpoint(**entry) for entry in payload["endpoints"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"endpoint document contains an invalid entry: {error}"
        ) from error
    if not endpoints:
        raise ValueError("endpoint document contains no endpoints")
    model_ids = [endpoint.model_id for endpoint in endpoints]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("endpoint document repeats a model id")
    return endpoints


def publish_endpoint_document(
    registrations: Sequence[ServeRegistration],
    path: str | Path,
    *,
    fetch_catalog: Callable[[str], object],
) -> Path:
    """Build and atomically write the current anonymous endpoint document."""
    return write_endpoint_document(
        build_endpoint_document(registrations, fetch_catalog=fetch_catalog), path
    )


def _fetch_anonymous_catalog(origin: str) -> object:
    """Fetch one engine catalog without credentials or ambient proxies."""
    import urllib.error
    import urllib.request

    class RejectRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, new_url):
            raise urllib.error.HTTPError(
                request.full_url,
                code,
                "endpoint redirect refused",
                headers,
                fp,
            )

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), RejectRedirects()
    )
    request = urllib.request.Request(f"{origin}/v1/models", method="GET")
    with opener.open(request, timeout=5) as response:
        if not 200 <= response.status < 300:
            raise OSError(f"endpoint returned HTTP {response.status}")
        return json.load(response)


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
    publish = subparsers.add_parser("publish")
    publish.add_argument("--directory")
    publish.add_argument("--output")
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
    elif args.command == "remove":
        remove_registration(args.job_id, args.directory)
    else:
        from imas_ambix.agent.profile import SiteConfig

        site = SiteConfig.from_env()
        directory = args.directory or registration_directory(site.base_dir)
        output = args.output or site.endpoint_document
        registrations = read_registrations(
            directory, job_is_running=lambda _job_id: True
        ).current
        publish_endpoint_document(
            registrations, output, fetch_catalog=_fetch_anonymous_catalog
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
