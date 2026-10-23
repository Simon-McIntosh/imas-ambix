"""Shared registration records for running model serves.

A registration is a route candidate, not availability evidence. Callers must
still reconcile job liveness and successfully probe the registered endpoint
before using it as an upstream.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from imas_ambix.agent.vllm_catalog import validate_catalog_metadata

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from imas_ambix.agent.profile import SiteServerConfig

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_ROUTER_COMMENT_PREFIX = "ambix-router;"


def _short_host(host: str) -> str:
    """Return a host's short name: the node name without its cluster component.

    The component is dropped only when what follows it still reads as a node
    name, which needs the second separator: ``98dci4-gpu-0003`` is
    ``gpu-0003``, while ``gpu-0003`` and ``node-a`` are already short names and
    come back unchanged.
    """
    parts = host.split("-")
    return "-".join(parts[1:]) if len(parts) >= 3 else host


def _engine_id(host: str, port: int) -> str:
    """Return an engine's identity: its host short name and its port."""
    return f"{_short_host(host)}:{port}"


def _server_index(servers: Sequence[SiteServerConfig]) -> dict[str, str]:
    """Map every spelling of a declared server onto its declared name.

    A registration may name its server by declaration name or carry only the
    host it ran on, and the host may be spelled with or without its cluster
    component, so all three are accepted. The first spelling of a name wins, so
    a declaration order decides between two servers that share one.
    """
    index: dict[str, str] = {}
    for server in servers:
        for key in (server.name, server.host, _short_host(server.host)):
            if key:
                index.setdefault(key, server.name)
    return index


def _server_key(registration: ServeRegistration, index: dict[str, str]) -> str:
    """Return the servers-array entry this engine belongs under.

    A registration that names a declared server is grouped under it. Anything
    else -- a record written before servers had names, or a name this site does
    not declare -- is grouped under an entry named by the host it ran on, so no
    engine is dropped from the document for want of a declaration.
    """
    for candidate in (registration.server, registration.host):
        if candidate and candidate in index:
            return index[candidate]
    return registration.host


@dataclass(frozen=True, slots=True)
class ServeRegistration:
    """Resolved identity written by one serving allocation."""

    model_id: str
    host: str
    port: int
    job_id: str
    accelerator_count: int
    checkpoint_precision: str
    accelerator_family: str
    server: str = ""
    held: bool = False

    @property
    def engine_id(self) -> str:
        """Identity of the engine this allocation serves: host short name and port."""
        return _engine_id(self.host, self.port)

    def __post_init__(self) -> None:
        if not isinstance(self.held, bool):
            raise ValueError("held must be a boolean")
        for name in (
            "model_id",
            "host",
            "job_id",
            "checkpoint_precision",
            "accelerator_family",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not _SAFE_JOB_ID.fullmatch(self.job_id):
            raise ValueError("job_id contains characters unsafe for a record name")
        # Absent server names are accepted: a record written before a site
        # declared its servers names no server, and reading it must not fail.
        if not isinstance(self.server, str):
            raise ValueError("server must be text")
        if any(
            ord(character) < 32 or ord(character) == 127 for character in self.server
        ):
            raise ValueError("server contains control characters")
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

    @property
    def engine_id(self) -> str:
        """Identity of this engine: host short name and port."""
        return _engine_id(self.host, self.port)


@dataclass(frozen=True, slots=True)
class PublishedServer:
    """One server and the engines on it, as the document reports them.

    It is the site's declaration rather than a summary of its engines: a
    declared server appears whether or not anything is running on it, with the
    accelerators and card count it declares, and an empty ``engines`` list says
    so. ``accelerator_count`` is ``None`` where the site declares no card count,
    which is what an unconfigured site's single entry does: the count is a
    property of the node, an engine is one serve placed on it, and a serve's own
    count is reported on that engine. ``power_cap_watts`` and
    ``reservation_groups`` are carried because they are declared here and read
    from this document, not from the site configuration.
    """

    name: str
    host: str
    accelerator_family: str
    accelerator_count: int | None
    power_cap_watts: int | None
    reservation_groups: tuple[str, ...]
    engines: tuple[str, ...]


class EndpointDocument(tuple):
    """The accepted endpoints, carrying the server grouping they were built from.

    It is a tuple of endpoints so every reader that wants only the engines is
    unchanged, and it carries the servers array beside them. One builder
    produces both, because a server grouping computed somewhere else would
    disagree with the engines it claims to group.
    """

    servers: tuple[PublishedServer, ...]

    def __new__(
        cls,
        endpoints: Sequence[PublishedEndpoint] = (),
        servers: Sequence[PublishedServer] = (),
    ) -> EndpointDocument:
        document = super().__new__(cls, endpoints)
        document.servers = tuple(servers)
        return document


@dataclass(frozen=True, slots=True)
class PublishedOrigin:
    """One anonymously reachable candidate for multi-release routing."""

    host: str
    port: int

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")
        if any(ord(character) < 32 or ord(character) == 127 for character in self.host):
            raise ValueError("host contains control characters")
        if not isinstance(self.port, int) or isinstance(self.port, bool):
            raise ValueError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @property
    def origin(self) -> str:
        """Direct HTTP origin for this routing candidate."""
        return f"http://{self.host}:{self.port}"


def registration_directory(base_dir: str | Path) -> Path:
    """Return the shared per-serve registry directory for a site base."""
    return Path(base_dir) / "agents" / "serve-registry"


def _record_path(directory: Path, job_id: str) -> Path:
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("job_id contains characters unsafe for a record name")
    return directory / f"{job_id}.json"


def write_registration(registration: ServeRegistration, directory: str | Path) -> Path:
    """Write launch identity without changing an existing admission decision."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _record_path(target_dir, registration.job_id)
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            previous = ServeRegistration(**json.loads(target.read_text()))
            registration = replace(registration, held=previous.held)
        return _write_registration(registration, target_dir)


def release_registration(job_id: str, directory: str | Path) -> Path:
    """Release a validated allocation; subsequent launch writes preserve release."""
    target_dir = Path(directory)
    target = _record_path(target_dir, job_id)
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registration = ServeRegistration(**json.loads(target.read_text()))
        if registration.job_id != job_id:
            raise ValueError("registration job id disagrees with its path")
        return _write_registration(replace(registration, held=False), target_dir)


def _write_registration(registration: ServeRegistration, target_dir: Path) -> Path:
    """Atomically publish a record while its caller holds the job's lock."""
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
    # Topology is taken from the registration, which the serving allocation
    # writes from its own profile. That is the same launch-owned source the
    # vLLM catalog middleware reads out of AMBIX_VLLM_CATALOG_METADATA before
    # echoing it back through /v1/models, so reading it here loses no authority
    # and is not a client-side inference -- it just stops requiring every
    # engine to carry our own data back to us. An engine with no way to do so
    # (SGLang takes no middleware) is then routable on the same terms.
    #
    # Where a card DOES carry the block, it still has to agree. A disagreement
    # means the registration and the endpoint describe different serves, which
    # is exactly the integrity failure the check exists to catch; absence means
    # only that the engine has no channel for it.
    card_metadata = item.get("ambix")
    if card_metadata is not None:
        metadata = validate_catalog_metadata({registration.model_id: card_metadata})[
            registration.model_id
        ]
        if metadata["accelerator_count"] != registration.accelerator_count:
            raise ValueError("endpoint accelerator count disagrees with registration")
        if metadata["checkpoint_precision"] != registration.checkpoint_precision:
            raise ValueError(
                "endpoint checkpoint precision disagrees with registration"
            )
        if metadata["accelerator_family"] != registration.accelerator_family:
            raise ValueError("endpoint accelerator family disagrees with registration")
    parsed = urlsplit(registration.origin)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError("registration origin is incomplete")
    return PublishedEndpoint(
        model_id=registration.model_id,
        host=parsed.hostname,
        port=parsed.port,
        accelerator_family=registration.accelerator_family,
        accelerator_count=registration.accelerator_count,
        checkpoint_precision=registration.checkpoint_precision,
        max_context=item.get("max_model_len"),
    )


def build_endpoint_document(
    registrations: Sequence[ServeRegistration],
    *,
    fetch_catalog: Callable[[str], object],
    servers: Sequence[SiteServerConfig] = (),
) -> EndpointDocument:
    """Probe registrations anonymously and retain complete live endpoints.

    Two engines serving one model both publish, because that is what a second
    server is for; what is deduplicated is the engine, so a restarted job on
    the same engine still replaces its predecessor.

    ``servers`` is the site's declaration, passed in by the caller rather than
    read from the configuration here: every declared server appears in the
    document's servers array whether or not an engine is running on it, and the
    running engines are grouped under the server their registration names.
    """
    endpoints: list[PublishedEndpoint] = []
    index = _server_index(servers)
    # engine id -> (endpoint, the job id that registered it, its server)
    seen_engines: dict[str, tuple[PublishedEndpoint, str, str]] = {}
    for registration in registrations:
        if registration.held:
            continue
        try:
            endpoint = _endpoint_from_catalog(
                registration, fetch_catalog(registration.origin)
            )
        except (OSError, TypeError, ValueError) as error:
            # Say why. Dropping a registration silently is how a whole routing
            # lane can go unserviceable while every symptom points elsewhere:
            # the document simply comes back short, callers see "no endpoints",
            # and the reason is nowhere. Measured 2026-09-14 -- six runs across
            # four sessions failed against a model id that looked wrong, when
            # the endpoint had been dropped here for missing metadata.
            _LOGGER.warning(
                "dropping registration %s (job %s) at %s: %s",
                registration.model_id,
                registration.job_id,
                registration.origin,
                error,
            )
            continue
        server = _server_key(registration, index)
        engine = endpoint.engine_id
        prior = seen_engines.get(engine)
        if prior is not None:
            previous, previous_job, previous_server = prior
            # Two registrations describing one engine is almost always a dead
            # job whose successor took the port it released: both probe the one
            # live serve, so both look current. Resolve to the later job rather
            # than refusing, and never abort the build -- raising here empties
            # the document for EVERY engine, so one stale file takes the whole
            # routing lane down while the serve it names is answering.
            newer = registration.job_id > previous_job
            keep = endpoint if newer else previous
            keep_job = registration.job_id if newer else previous_job
            keep_server = server if newer else previous_server
            _LOGGER.warning(
                "duplicate registrations publish engine %s at %s and %s; "
                "keeping job %s",
                engine,
                previous.origin,
                endpoint.origin,
                keep_job,
            )
            endpoints[endpoints.index(previous)] = keep
            seen_engines[engine] = (keep, keep_job, keep_server)
            continue
        endpoints.append(endpoint)
        seen_engines[engine] = (endpoint, registration.job_id, server)
    ordered = tuple(
        sorted(endpoints, key=lambda endpoint: (endpoint.model_id, endpoint.engine_id))
    )
    return EndpointDocument(
        ordered, _published_servers(ordered, seen_engines, servers)
    )


def _published_servers(
    endpoints: Sequence[PublishedEndpoint],
    seen_engines: dict[str, tuple[PublishedEndpoint, str, str]],
    servers: Sequence[SiteServerConfig],
) -> tuple[PublishedServer, ...]:
    """Report every declared server, with the engines grouped under each.

    The declaration is the authority for a declared server's host, accelerator
    family and card count: a serve may take fewer cards than its node has, so
    an engine is not a reading of the node it places on, and neither is any
    other engine. A card count is published only where the site declares one;
    otherwise it is unknown, and an engine's own count stays on that engine's
    endpoint rather than being promoted to a statement about the node.

    A server the site does not declare still appears, under the host its
    registration ran on, so an engine is never dropped from the document for
    want of a declaration. Declared servers keep declaration order and the
    remaining entries follow, sorted by name.
    """
    grouped: dict[str, list[PublishedEndpoint]] = {}
    for endpoint in endpoints:
        grouped.setdefault(seen_engines[endpoint.engine_id][2], []).append(endpoint)
    published: list[PublishedServer] = []
    for server in servers:
        engines = sorted(grouped.get(server.name, ()), key=lambda item: item.engine_id)
        published.append(
            PublishedServer(
                name=server.name,
                host=server.host,
                accelerator_family=server.accelerator_family,
                accelerator_count=server.accelerator_count,
                power_cap_watts=server.power_cap_watts,
                reservation_groups=tuple(server.reservation_groups),
                engines=tuple(endpoint.engine_id for endpoint in engines),
            )
        )
    declared_names = {server.name for server in servers}
    for name in sorted(set(grouped) - declared_names):
        engines = sorted(grouped[name], key=lambda item: item.engine_id)
        first = engines[0]
        published.append(
            PublishedServer(
                name=name,
                host=first.host,
                accelerator_family=first.accelerator_family,
                accelerator_count=None,
                power_cap_watts=None,
                reservation_groups=(),
                engines=tuple(endpoint.engine_id for endpoint in engines),
            )
        )
    return tuple(published)


def write_endpoint_document(
    endpoints: Sequence[PublishedEndpoint],
    path: str | Path,
    *,
    routing_origins: Sequence[PublishedOrigin] = (),
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
        {
            "endpoints": [asdict(endpoint) for endpoint in endpoints],
            "routing_origins": [asdict(origin) for origin in routing_origins],
            "servers": [
                asdict(server) for server in getattr(endpoints, "servers", ())
            ],
        },
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
    # Engines are unique; model ids are not, because a pool of more than one
    # engine repeats its model id by design.
    engine_ids = [endpoint.engine_id for endpoint in endpoints]
    if len(engine_ids) != len(set(engine_ids)):
        raise ValueError("endpoint document repeats an engine")
    return endpoints


def publish_endpoint_document(
    registrations: Sequence[ServeRegistration],
    path: str | Path,
    *,
    fetch_catalog: Callable[[str], object],
    servers: Sequence[SiteServerConfig] = (),
    routing_origins: Sequence[PublishedOrigin] = (),
) -> Path:
    """Build and atomically write the current anonymous endpoint document."""
    return write_endpoint_document(
        build_endpoint_document(
            registrations, fetch_catalog=fetch_catalog, servers=servers
        ),
        path,
        routing_origins=routing_origins,
    )


def discover_routing_origins(
    endpoints: Sequence[PublishedEndpoint], jobs: Sequence[dict[str, str]]
) -> tuple[PublishedOrigin, ...]:
    """Qualify scheduler-discovered routers against every published release."""
    from imas_ambix.agent.cli import (
        _SERVE_COMMENT_PREFIX,
        _job_node,
        _probe_endpoint,
        _serve_port,
    )

    required_models = {endpoint.model_id for endpoint in endpoints}
    if not required_models:
        return ()
    origins: list[PublishedOrigin] = []
    seen_origins: set[str] = set()
    for job in jobs:
        comment = job.get("comment", "")
        if job.get("state") != "RUNNING" or not comment.startswith(
            _ROUTER_COMMENT_PREFIX
        ):
            continue
        node = _job_node(job)
        port = _serve_port(
            _SERVE_COMMENT_PREFIX + comment.removeprefix(_ROUTER_COMMENT_PREFIX)
        )
        if node is None or port is None:
            continue
        origin = f"http://{node}:{port}"
        if origin in seen_origins:
            continue
        probe = _probe_endpoint(origin, None)
        if probe.readiness != "ready":
            continue
        available_models = {model.model_id for model in probe.models}
        if not required_models.issubset(available_models):
            continue
        origins.append(PublishedOrigin(host=node, port=port))
        seen_origins.add(origin)
    return tuple(sorted(origins, key=lambda item: (item.host, item.port)))


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
    write.add_argument("--accelerator-family", required=True)
    write.add_argument("--server", default="")
    write.add_argument("--held", action="store_true")
    release = subparsers.add_parser("release")
    release.add_argument("--directory", required=True)
    release.add_argument("--job-id", required=True)
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
                accelerator_family=args.accelerator_family,
                server=args.server,
                held=args.held,
            ),
            args.directory,
        )
    elif args.command == "release":
        release_registration(args.job_id, args.directory)
    elif args.command == "remove":
        remove_registration(args.job_id, args.directory)
    else:
        from imas_ambix.agent.cli import _running_jobs
        from imas_ambix.agent.profile import SiteConfig

        site = SiteConfig.from_env()
        directory = args.directory or registration_directory(site.base_dir)
        output = args.output or site.endpoint_document
        # Ask the scheduler which jobs are alive rather than assuming all of
        # them are. A registration outliving its job is how the published
        # catalog comes to name a serve that is gone, or to collide with the
        # successor that took its port. If the query itself fails it returns
        # nothing, which must not be read as "every job is dead" -- that would
        # publish an empty document over a healthy lane, so fall back to
        # retaining the records and let the origin probe decide.
        live = {job["jobid"] for job in _running_jobs(site)}
        job_is_running = (lambda job_id: job_id in live) if live else (lambda _: True)
        registrations = read_registrations(
            directory, job_is_running=job_is_running
        ).current
        endpoints = build_endpoint_document(
            registrations,
            fetch_catalog=_fetch_anonymous_catalog,
            servers=site.declared_servers,
        )
        write_endpoint_document(
            endpoints,
            output,
            routing_origins=discover_routing_origins(endpoints, _running_jobs(site)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
