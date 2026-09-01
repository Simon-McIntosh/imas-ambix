"""CLI commands for managing Ambix agent deployments."""

from __future__ import annotations

import getpass
import os
import re
import secrets
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from imas_ambix.agent.profile import SiteConfig, list_profiles, load_profile

if TYPE_CHECKING:
    from imas_ambix.agent.bench import BenchReport
    from imas_ambix.agent.router import Upstream

console = Console()

# Engine types that have uv-managed environments
ENGINE_TYPES = ("vllm", "sglang")

_SERVE_COMMENT_PREFIX = "ambix-serve;"
_SUPPORTED_GPU_COUNTS = frozenset({2, 4, 6, 8})


@dataclass(frozen=True)
class ModelMetadata:
    """Model identity and optional context reported by a live endpoint."""

    model_id: str
    max_context: int | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Sanitized result of probing an OpenAI-compatible models endpoint."""

    readiness: str
    models: tuple[ModelMetadata, ...] = ()


@dataclass(frozen=True)
class LiveRoute:
    """A scheduler candidate qualified by its own models endpoint."""

    model_id: str
    node: str
    port: int
    gpu_count: int
    job_id: str
    base_url: str
    max_context: int | None
    readiness: str
    job_name: str

    @property
    def topology(self) -> str:
        """Human-readable launch topology."""
        return f"{self.gpu_count}×H200" if self.gpu_count else "override"

    @property
    def selector(self) -> str:
        """Canonical route selector, unique across concurrent allocations."""
        return f"{self.model_id}@{self.gpu_count}xh200#{self.job_id}"

    @property
    def label(self) -> str:
        """Credential-free display label."""
        return (
            f"{self.model_id} · {self.topology} · "
            f"{self.node}:{self.port} · job {self.job_id}"
        )


# ── Config resolution ───────────────────────────────────────────────


def _load_dotenv() -> dict[str, str]:
    """Load KEY=VALUE pairs from a ``.env`` file if present.

    Supports comments (``#``), blank lines, optional quoting, and
    does **not** override variables already set in the environment.
    """
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            values[key] = val
            os.environ[key] = val
    return values


def _read_key_file(path: Path) -> str | None:
    """Read ``AMBIX_AGENT_API_KEY`` from a dotenv-style file."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == "AMBIX_AGENT_API_KEY":
            return val.strip().strip("\"'") or None
    return None


def _resolve_api_key(cli_value: str | None) -> str | None:
    """Resolve API key: CLI flag > envvar > CWD .env > shared agents/.env."""
    if cli_value:
        return cli_value
    env_key = os.environ.get("AMBIX_AGENT_API_KEY")
    if env_key:
        return env_key
    _load_dotenv()
    env_key = os.environ.get("AMBIX_AGENT_API_KEY")
    if env_key:
        return env_key
    site = SiteConfig.from_env()
    return _read_key_file(site.api_key_file)


def _resolve_serve_auth(auth: bool, cli_value: str | None, site) -> str | None:
    """Resolve the key a serve should enforce, or ``None`` for an open port.

    An open endpoint is the default: the cluster is already the authentication
    boundary, and a key readable only inside one storage group would lock the
    standalone consumer launcher out of the endpoint it exists to reach.
    Naming a key implies wanting it enforced, so a value arms auth on its own
    rather than being silently discarded, and asking for auth with no
    resolvable key is a launch error rather than a quiet downgrade to an open
    port -- an operator who asked to be protected must not be handed the
    opposite.
    """
    if not (auth or cli_value):
        return None
    resolved = _resolve_api_key(cli_value)
    if resolved is None:
        raise click.ClickException(
            "--auth was requested but no API key resolved from --api-key, "
            f"AMBIX_AGENT_API_KEY, or {site.api_key_file}."
        )
    return resolved


def _agent_config() -> dict[str, str]:
    """Resolve ``[tool.ambix.agent]`` from nearest pyproject.toml."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            import tomllib

            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                return data.get("tool", {}).get("ambix", {}).get("agent", {})
            except Exception:
                return {}
    return {}


def _default_profile() -> str | None:
    """Resolve default profile: envvar > pyproject.toml [tool.ambix.agent]."""
    env_val = os.environ.get("AMBIX_AGENT_DEFAULT_PROFILE")
    if env_val:
        return env_val
    return _agent_config().get("default_profile")


def _default_url() -> str | None:
    """Resolve default URL: envvar > pyproject.toml [tool.ambix.agent]."""
    env_val = os.environ.get("AMBIX_AGENT_URL")
    if env_val:
        return env_val
    return _agent_config().get("url")


def _resolve_slug(slug: str | None) -> str:
    """Return *slug* if given, else fall back to default profile."""
    if slug:
        return slug
    default = _default_profile()
    if default:
        return default
    raise click.ClickException(
        "No profile specified and no default_profile configured.\n"
        "Set [tool.ambix.agent] default_profile in pyproject.toml "
        "or AMBIX_AGENT_DEFAULT_PROFILE envvar."
    )


def _load_profile(slug: str | None):
    resolved = _resolve_slug(slug)
    try:
        return load_profile(resolved)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


def _scale_profile(profile, gpus: int):
    """Return a copy of *profile* with GPU count and dependent resources scaled.

    Adjusts ``engine.tensor_parallel``, ``slurm.gpus`` and ``slurm.memory``.
    Host memory scales because it stages weights, whose footprint follows the
    card count; ``slurm.cpus`` deliberately does NOT, because an engine's cores
    run one API server, one engine core and one worker per rank, and every
    worker spends most of its time blocked on the device. Scaling them off cards
    exhausts a reservation that two groups share and leaves a co-running job
    pending on cores beside idle GPUs. Override a single launch with ``--cpus``.
    The caller's profile is not mutated.
    """
    # A declared variant for this card count carries its own checkpoint and
    # sizing, so it replaces proportional scaling rather than being scaled.
    variant = profile.for_gpus(gpus)
    if variant is not profile:
        return variant
    base_gpus = profile.slurm.gpus
    if gpus == base_gpus:
        return profile
    ratio = gpus / base_gpus
    mem_str = profile.slurm.memory
    mem_val = int(mem_str[:-1])
    mem_unit = mem_str[-1]
    new_memory = f"{max(1, round(mem_val * ratio))}{mem_unit}"
    return profile.model_copy(
        deep=True,
        update={
            "slurm": profile.slurm.model_copy(
                update={"gpus": gpus, "memory": new_memory}
            ),
            "engine": profile.engine.model_copy(update={"tensor_parallel": gpus}),
        },
    )


@click.group()
def agent() -> None:
    """Manage LLM agent deployments on SLURM GPU clusters."""


@agent.command(name="list")
def list_command() -> None:
    """List available model profiles, marking any that are serving."""
    site = SiteConfig.from_env()
    serving = _serving_slugs(site)

    console.print("[bold]imas-ambix profiles[/]")
    table = Table(box=None, pad_edge=False)
    table.add_column("SLUG", style="cyan", no_wrap=True)
    table.add_column("MODEL", style="bold")
    table.add_column("ENGINE")
    table.add_column("SIZE", justify="right")
    table.add_column("STATUS")

    for slug in list_profiles():
        profile = _load_profile(slug)
        status_cell = "[green]serving[/]" if slug in serving else ""
        table.add_row(
            profile.slug,
            profile.model.name,
            _ENGINE_LABELS.get(profile.engine.type, profile.engine.type),
            f"{profile.model.size_gb} GB",
            status_cell,
        )

    console.print(table)


@agent.command()
@click.argument("slug", required=False, default=None)
def info(slug: str | None) -> None:
    """Show detailed information for a model profile."""
    profile = _load_profile(slug)
    site = SiteConfig.from_env()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("Model name", profile.model.name)
    table.add_row("HF repo", profile.model.hf_repo)
    table.add_row("Served name", profile.model.served_name)
    table.add_row(
        "Engine", _ENGINE_LABELS.get(profile.engine.type, profile.engine.type)
    )
    table.add_row("Tensor parallel", str(profile.engine.tensor_parallel))
    table.add_row("Attention backend", profile.engine.attention_backend)
    table.add_row("Max context", f"{profile.model.max_context:,}")
    table.add_row("Model size", f"{profile.model.size_gb} GB")
    table.add_row(
        "SLURM",
        f"{profile.slurm.gpus}×H200 · {profile.slurm.cpus} CPU "
        f"· {profile.slurm.memory} · serve {profile.slurm.time_serve}",
    )
    table.add_row("Model directory", str(site.model_dir(profile)))
    table.add_row("Cache directory", str(site.cache_dir(profile)))
    if profile.engine.ktransformers is not None:
        table.add_row(
            "KTransformers",
            "\n".join(
                f"{key}: {value}"
                for key, value in profile.engine.ktransformers.model_dump().items()
            ),
        )
    parsers = {
        key: value
        for key, value in profile.engine.parsers.model_dump().items()
        if value is not None
    }
    if parsers:
        table.add_row(
            "Parsers",
            "\n".join(f"{key}: {value}" for key, value in parsers.items()),
        )
    console.print(
        Panel(
            table,
            title=f"[bold]{profile.slug}[/] → {profile.model.served_name}",
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        )
    )


@agent.command()
@click.argument("slug", required=False, default=None)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SLURM script instead of submitting it.",
)
def download(slug: str | None, dry_run: bool) -> None:
    """Generate and submit a model download job."""
    from imas_ambix.agent.slurm import generate_download_script, submit_script

    profile = _load_profile(slug)
    site = SiteConfig.from_env()
    script = generate_download_script(profile, site)

    if dry_run:
        console.print(script, markup=False, highlight=False, soft_wrap=True)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"Submitted download job {job_id} for {profile.slug}.")


@agent.command()
@click.argument("slug", required=False, default=None)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SLURM script instead of submitting it.",
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=None,
    help="Port to expose on the compute node.",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="API key for authenticating client requests (also AMBIX_AGENT_API_KEY).",
)
@click.option(
    "--auth",
    "auth",
    is_flag=True,
    help="Require an API key on /v1/* requests, resolved from --api-key, "
    "AMBIX_AGENT_API_KEY, or the shared key file. The default is an open "
    "endpoint: the cluster is already the authentication boundary, and a key "
    "readable only inside one storage group would lock consumers out of it.",
)
@click.option(
    "--gpus",
    type=int,
    default=None,
    help="Override number of GPUs (and tensor-parallel size). "
    "Scales memory proportionally from the profile default. Host cores do "
    "not scale with cards; use --cpus to override them.",
)
@click.option(
    "--cpus",
    type=click.IntRange(1, 64),
    default=None,
    help="Override host cores, applied after any --gpus scaling. This node "
    "carries two overlapping 30-core reservations, so cores are the scarce "
    "resource a serve competes for: an inference server spends most of its "
    "time blocked on the device, and the proportional default reserves cores "
    "a co-running job could use. Probe with `srun --test-only` before "
    "committing to a value.",
)
@click.option(
    "--time",
    "time_limit",
    default=None,
    help="SLURM walltime for this serve, e.g. 3:00:00. Declaring far more than "
    "the work needs makes the scheduler plan the node as occupied for that "
    "long, so short jobs from other users pend behind it. Set it for a "
    "measurement run; the profile default suits a persistent service.",
)
@click.option(
    "--no-speculative",
    is_flag=True,
    help="Serve without speculative decoding. Drafting only pays for itself "
    "when acceptance is high and the batch is large enough to amortise the "
    "draft pass; below that it costs decode rate, so measure both ways.",
)
def serve(
    slug: str | None,
    dry_run: bool,
    port: int | None,
    api_key: str | None,
    auth: bool,
    gpus: int | None,
    cpus: int | None,
    no_speculative: bool,
    time_limit: str | None,
) -> None:
    """Generate and submit a model serving job."""
    from imas_ambix.agent.slurm import generate_serve_script, submit_script

    profile = _load_profile(slug)
    if gpus is not None:
        profile = _scale_profile(profile, gpus)
    if cpus is not None:
        profile = profile.model_copy(
            update={"slurm": profile.slurm.model_copy(update={"cpus": cpus})}
        )
    if time_limit:
        profile = profile.model_copy(
            update={
                "slurm": profile.slurm.model_copy(update={"time_serve": time_limit})
            }
        )
    if no_speculative:
        profile = profile.model_copy(
            update={
                "engine": profile.engine.model_copy(
                    update={
                        "speculative_method": None,
                        "speculative_num_tokens": None,
                        "speculative_model": None,
                    }
                )
            }
        )
    site = SiteConfig.from_env()
    resolved_port = port if port is not None else site.default_port
    resolved_key = _resolve_serve_auth(auth, api_key, site)
    script = generate_serve_script(
        profile, site, port=resolved_port, api_key=resolved_key
    )

    if dry_run:
        if resolved_key:
            script = script.replace(resolved_key, "****")
        console.print(script, markup=False, highlight=False, soft_wrap=True)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    key_note = " (API key enabled)" if resolved_key else " (NO AUTH — open endpoint)"
    gpu_note = (
        f" ({profile.slurm.gpus}×GPU, {profile.slurm.cpus} CPU)"
        if gpus is not None or cpus is not None
        else ""
    )
    message = (
        f"Submitted serve job {job_id} for {profile.slug}{gpu_note} "
        f"on port {resolved_port}{key_note}."
    )
    console.print(message)


def _job_node(job: dict[str, str]) -> str | None:
    """Return a trustworthy compute-node host for a running allocation."""
    node = job.get("node", "")
    return node if node and "(" not in node else None


@agent.command()
@click.option("--reveal", is_flag=True, help="Show the full API key in plaintext.")
def status(reveal: bool) -> None:
    """Show active Ambix agent jobs and how to connect to running models.

    Prints a compact job table, then a connection detail block for each
    RUNNING serve job: served model name, the client URL, a live
    ``/v1/models`` readiness probe, the API key, and the engine/model facts.

    Login and standard compute nodes route directly to the GPU node's serve
    port, so the URL is ``http://<gpu-node>:<port>`` and the probe is
    reliable from there. (SSH port-forwarding to the compute node is
    administratively prohibited, so there is no tunnel.)

    Each endpoint is probed unauthenticated to establish whether it enforces
    a key, because an open endpoint is the serve default and the shared key
    file existing says nothing about what a given port requires. Where a key
    IS enforced it is read from the shared mode-640 ``agents/.env``; if you
    lack read permission it shows ``(no access)`` — the filesystem is the gate.
    """
    site = SiteConfig.from_env()
    jobs = _running_jobs(site)

    if not jobs:
        console.print("No imas-ambix agent jobs found.")
        return

    try:
        key = _read_key_file(site.api_key_file)
        key_display = (
            (key if reveal else _mask_key(key)) if key else "(none configured)"
        )
    except PermissionError:
        key = None
        key_display = "(no access — not key owner)"

    routes, rejected = _discover_live_routes(site, key, jobs=jobs)
    live_job_ids = {route.job_id for route in routes}
    pending = [job for job in jobs if job["jobid"] not in live_job_ids]

    # -- Header line -------------------------------------------------------
    summary = f"{len(routes)} serving"
    if pending:
        summary += f" · {len(pending)} other"
    console.print(
        f"[bold]imas-ambix agents[/]  ·  {site.partition}    [dim]{summary}[/]"
    )

    # -- One boxed panel per probe-qualified route -------------------------
    jobs_by_id = {job["jobid"]: job for job in jobs}
    for route in routes:
        job = jobs_by_id[route.job_id]
        try:
            profile = load_profile(route.job_name)
            # Resolve to the card count this allocation actually holds. The
            # base profile carries its own default sizing, so reporting it
            # unresolved shows a tensor-parallel width and memory budget the
            # running engine never used.
            if route.gpu_count:
                profile = _scale_profile(profile, route.gpu_count)
            facts = _engine_facts(profile)
        except FileNotFoundError:
            facts = ""
        compute = f"{route.node} · {route.topology}"

        requires_key = _endpoint_requires_key(route.base_url)
        if requires_key is False:
            route_key_display = "(none — open endpoint)"
        elif requires_key is None:
            route_key_display = f"{key_display} [dim](unverified)[/]"
        else:
            route_key_display = key_display

        body = Table.grid(padding=(0, 2))
        body.add_column(style="cyan", no_wrap=True)
        body.add_column()
        body.add_row("URL", route.base_url)
        body.add_row("Key", route_key_display)
        if facts:
            body.add_row("Engine", facts)
        body.add_row("Compute", compute)
        if route.max_context is not None:
            body.add_row("Reported context", _fmt_context(route.max_context))
        body.add_row("Selector", route.selector)

        title = (
            f"[bold]{route.job_name}[/] → {route.model_id}   "
            f"[green]RUNNING[/] {_fmt_uptime(job['time'])} · job {job['jobid']}"
        )
        console.print()
        console.print(
            Panel(
                body,
                title=title,
                title_align="left",
                subtitle="[green]READY[/]",
                subtitle_align="right",
                border_style="green",
                padding=(0, 1),
            )
        )

    if rejected:
        console.print()
        console.print("[dim]rejected serve candidates[/]")
        for reason in rejected:
            console.print(f"  [yellow]{reason}[/]")

    # -- Pending / non-serve jobs (download, setup, queued) ----------------
    if pending:
        ptable = Table.grid(padding=(0, 2))
        ptable.add_column(style="cyan", no_wrap=True)
        ptable.add_column(style="bold")
        ptable.add_column()
        ptable.add_column()
        for job in pending:
            ptable.add_row(
                job["jobid"],
                job["name"],
                f"[yellow]{job['state']}[/]",
                job["node"],
            )
        console.print()
        console.print("[dim]other jobs[/]")
        console.print(ptable)


@agent.command()
@click.argument("slug", required=False, default=None)
@click.option("--all", "cancel_all", is_flag=True, help="Cancel ALL ambix jobs.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def shutdown(slug: str | None, cancel_all: bool, yes: bool) -> None:
    """Cancel active Ambix agent SLURM jobs.

    Without arguments, cancels serve jobs for the default profile.
    With SLUG, cancels serve jobs for that profile.
    With --all, cancels all ambix jobs (serve, download, setup).
    """
    user = os.environ.get("USER") or getpass.getuser()
    from imas_ambix.agent.profile import SiteConfig

    site = SiteConfig.from_env()
    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-A", site.account, "-o", "%i|%j"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "Failed to query squeue"
        raise click.ClickException(message)

    # Parse structured output
    jobs: list[tuple[str, str]] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        job_id, job_name = line.split("|", 1)
        job_id = job_id.strip()
        job_name = job_name.strip()
        jobs.append((job_id, job_name))

    if not jobs:
        console.print("No imas-ambix agent jobs found.")
        return

    # Filter by scope
    if cancel_all:
        targets = jobs
    else:
        resolved = _resolve_slug(slug) if slug else _default_profile()
        if resolved:
            targets = [(jid, jn) for jid, jn in jobs if jn == resolved]
        else:
            # No slug and no default: match known profile slugs only.
            known = set(list_profiles())
            targets = [(jid, jn) for jid, jn in jobs if jn in known]

    if not targets:
        console.print("No matching jobs to cancel.")
        return

    # Show what will be cancelled
    console.print("[bold]Jobs to cancel:[/]")
    for job_id, job_name in targets:
        console.print(f"  {job_id}  {job_name}")

    if not yes and not click.confirm("Proceed?"):
        console.print("Aborted.")
        return

    # Cancel jobs
    job_ids = [jid for jid, _ in targets]
    cancel_result = subprocess.run(
        ["scancel"] + job_ids,
        capture_output=True,
        text=True,
        check=False,
    )
    if cancel_result.returncode != 0:
        raise click.ClickException(cancel_result.stderr.strip() or "scancel failed")
    console.print(f"[green]Cancelled {len(job_ids)} job(s).[/]")


# -- Key management ----------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked representation of an API key."""
    if len(key) <= 8:
        return key[:2] + "***"
    return key[:4] + "..." + key[-4:]


def _running_jobs(
    site: SiteConfig, *, job_ids: tuple[str, ...] = ()
) -> list[dict[str, str]]:
    """Return Ambix SLURM jobs as a list of field dicts.

    Each entry has ``jobid``, ``name`` (the profile slug), ``state``,
    ``time``, ``node`` (or the pending reason), allocated ``gres``, and the
    scheduler ``comment``. With explicit ids the query reconciles shared
    registrations independent of their owner; otherwise it retains the
    operator-scoped status view. Query failure is distinct from an empty queue.
    """
    selector = (
        ["-j", ",".join(job_ids)]
        if job_ids
        else [
            "-u",
            os.environ.get("USER") or getpass.getuser(),
        ]
    )
    result = subprocess.run(
        [
            "squeue",
            "-h",
            *selector,
            "-A",
            site.account,
            "-o",
            "%i|%j|%T|%M|%R|%b|%k",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(result.stderr.strip() or "Failed to query squeue")
    jobs: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        jobs.append(
            {
                "jobid": parts[0].strip(),
                "name": parts[1].strip(),
                "state": parts[2].strip(),
                "time": parts[3].strip(),
                "node": parts[4].strip(),
                "gres": parts[5].strip(),
                "comment": parts[6].strip(),
            }
        )
    return jobs


def _serving_slugs(site: SiteConfig) -> set[str]:
    """Profile slugs whose running allocation passed its endpoint probe."""
    known = set(list_profiles())
    routes, _ = _discover_live_routes(site, _read_key_file(site.api_key_file))
    return {route.job_name for route in routes if route.job_name in known}


def _endpoint_requires_key(url: str, timeout: float = 4.0) -> bool | None:
    """Whether ``{url}/v1/models`` rejects an unauthenticated request.

    The serve default is an open endpoint, so the shared key file existing says
    nothing about what a given endpoint enforces. Reporting the file's key
    beside an open port tells an operator the opposite of the truth, and the
    endpoint can answer the question directly. ``None`` means the probe itself
    was inconclusive.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(f"{url.rstrip('/')}/v1/models")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status != 200
    except urllib.error.HTTPError as error:
        return error.code in {401, 403}
    except OSError:
        return None


def _probe_endpoint(url: str, api_key: str | None, timeout: float = 4.0) -> ProbeResult:
    """Probe ``{url}/v1/models`` and return validated, sanitized metadata."""
    import json as json_module
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"{url.rstrip('/')}/v1/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return ProbeResult(f"http {resp.status}")
            try:
                payload = json_module.load(resp)
            except (
                json_module.JSONDecodeError,
                UnicodeDecodeError,
                TypeError,
            ):
                return ProbeResult("malformed response")
            raw_models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(raw_models, list) or not raw_models:
                return ProbeResult("empty models")
            models: list[ModelMetadata] = []
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    continue
                if any(ord(character) < 32 for character in model_id):
                    continue
                context = item.get("max_model_len", item.get("max_context_length"))
                if not isinstance(context, int) or isinstance(context, bool):
                    context = None
                models.append(ModelMetadata(model_id.strip(), context))
            if not models:
                return ProbeResult("empty models")
            return ProbeResult("ready", tuple(models))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return ProbeResult("auth-fail")
        return ProbeResult(f"http {exc.code}")
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ):
        return ProbeResult("unreachable")


def _serve_port(comment: str) -> int | None:
    """Extract a validated port from an Ambix scheduler comment."""
    if not comment.startswith(_SERVE_COMMENT_PREFIX):
        return None
    fields = dict(
        field.split("=", 1)
        for field in comment[len(_SERVE_COMMENT_PREFIX) :].split(";")
        if "=" in field
    )
    try:
        port = int(fields.get("port", ""))
    except ValueError:
        return None
    return port if 1 <= port <= 65535 else None


def _batch_script_port(job_id: str) -> int | None:
    """Recover one concrete serve port from scheduler-owned batch metadata."""
    result = subprocess.run(
        ["scontrol", "write", "batch_script", job_id, "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    pattern = re.compile(
        r"^\s*PORT=(?:['\"]?(\d+)['\"]?|\$\{AMBIX_PORT:-(\d+)\})\s*$",
        re.MULTILINE,
    )
    ports = {
        int(value)
        for match in pattern.finditer(result.stdout)
        for value in match.groups()
        if value is not None and 1 <= int(value) <= 65535
    }
    return ports.pop() if len(ports) == 1 else None


def _allocated_gpus(gres: str) -> int | None:
    """Extract the GPU count from SLURM's allocated generic resources.

    The scheduler spells the same allocation several ways, and the resource
    name is not always the first token: ``squeue %b`` reports the TRES form
    ``gres/gpu:2`` and the typed form ``gres/gpu:h200:1``, while a plain
    ``gpu:2`` and the ``gres/gpu=2`` of a TRES list also occur. Anchoring on
    the bare name alone silently reads every prefixed form as no allocation,
    which presents a healthy endpoint as an unsupported one.
    """
    match = re.search(r"(?:^|,)(?:gres/)?gpu(?::[^,:=]+)?[:=](\d+)(?:\(|,|$)", gres)
    return int(match.group(1)) if match else None


def _discover_live_routes(
    site: SiteConfig,
    api_key: str | None,
    *,
    jobs: list[dict[str, str]] | None = None,
) -> tuple[list[LiveRoute], list[str]]:
    """Probe scheduler candidates and return only route-local ready models."""
    candidates = _running_jobs(site) if jobs is None else jobs
    routes: list[LiveRoute] = []
    rejected: list[str] = []
    known_profiles = set(list_profiles())
    for job in candidates:
        if job.get("state") != "RUNNING":
            continue
        comment = job.get("comment", "")
        job_name = job.get("name", "")
        if comment.startswith(_SERVE_COMMENT_PREFIX):
            port = _serve_port(comment)
        elif job_name in known_profiles:
            port = _batch_script_port(job.get("jobid", ""))
        else:
            continue
        job_id = job.get("jobid", "?")
        node = _job_node(job)
        gpu_count = _allocated_gpus(job.get("gres", ""))
        if node is None:
            rejected.append(f"job {job_id}: no allocated compute node")
            continue
        if port is None:
            rejected.append(f"job {job_id}: no trustworthy serve port")
            continue
        if gpu_count not in _SUPPORTED_GPU_COUNTS:
            rejected.append(f"job {job_id}: unsupported or missing GPU allocation")
            continue
        base_url = f"http://{node}:{port}"
        probe = _probe_endpoint(base_url, api_key)
        if probe.readiness != "ready":
            rejected.append(f"job {job_id}: {probe.readiness}")
            continue
        for model in probe.models:
            routes.append(
                LiveRoute(
                    model_id=model.model_id,
                    node=node,
                    port=port,
                    gpu_count=gpu_count,
                    job_id=job_id,
                    base_url=base_url,
                    max_context=model.max_context,
                    readiness=probe.readiness,
                    job_name=job_name,
                )
            )
    routes.sort(key=lambda route: (route.model_id, route.gpu_count, route.job_id))
    return routes, rejected


def _router_auth_header(api_key: str | None) -> tuple[str, str] | None:
    return ("Authorization", f"Bearer {api_key}") if api_key else None


def _resolve_router_upstreams(site: SiteConfig, api_key: str | None) -> list[Upstream]:
    """Prefer probe-qualified registrations, then add scheduler-only routes."""
    from imas_ambix.agent.registry import read_registrations, registration_directory
    from imas_ambix.agent.router import Upstream

    directory = registration_directory(site.base_dir)
    records = read_registrations(directory, job_is_running=lambda _job_id: True).current
    running_ids: set[str] | None = None
    if records:
        try:
            jobs = _running_jobs(
                site, job_ids=tuple(record.job_id for record in records)
            )
        except click.ClickException:
            # A scheduler outage must not erase still-probeable shared records.
            running_ids = None
        else:
            running_ids = {
                job["jobid"] for job in jobs if job.get("state") == "RUNNING"
            }

    auth_header = _router_auth_header(api_key)
    upstreams: list[Upstream] = []
    seen_origins: set[str] = set()
    registered_job_ids: set[str] = set()
    for record in records:
        if running_ids is not None and record.job_id not in running_ids:
            continue
        probe = _probe_endpoint(record.origin, None)
        if probe.readiness != "ready" or record.model_id not in {
            model.model_id for model in probe.models
        }:
            continue
        upstreams.append(
            Upstream(
                base_url=record.origin,
                auth_header=auth_header,
                model_id=record.model_id,
            )
        )
        seen_origins.add(record.origin.rstrip("/"))
        registered_job_ids.add(record.job_id)

    try:
        scheduler_routes, _ = _discover_live_routes(site, api_key)
    except click.ClickException:
        if upstreams:
            return upstreams
        raise
    for route in scheduler_routes:
        normalized_origin = route.base_url.rstrip("/")
        if route.job_id in registered_job_ids or normalized_origin in seen_origins:
            continue
        upstreams.append(
            Upstream(
                base_url=route.base_url,
                auth_header=auth_header,
                model_id=route.model_id,
            )
        )
        seen_origins.add(normalized_origin)
    return upstreams


@agent.command(name="router")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), required=True)
@click.option(
    "--api-key",
    default=None,
    help="Optional key forwarded to authenticated upstream engines.",
)
def router_command(host: str, port: int, api_key: str | None) -> None:
    """Serve the local multi-engine pass-through router."""
    from imas_ambix.agent.router import DynamicUpstreamResolver, serve_router

    site = SiteConfig.from_env()
    resolved_key = _resolve_api_key(api_key) if api_key else None
    resolver = DynamicUpstreamResolver(
        lambda: _resolve_router_upstreams(site, resolved_key)
    )
    serve_router(resolver, host=host, port=port)


def _resolve_live_route(
    routes: list[LiveRoute],
    selector: str | None,
    *,
    interactive: bool,
) -> LiveRoute:
    """Resolve a live route without silently choosing among ambiguous matches."""
    matches = routes
    if selector:
        folded = selector.casefold()
        matches = [
            route
            for route in routes
            if selector in {route.selector, route.job_id}
            or folded
            in {
                route.model_id.casefold(),
                f"{route.model_id}@{route.gpu_count}xh200".casefold(),
            }
        ]
    if not matches:
        detail = f" matching {selector!r}" if selector else ""
        raise click.ClickException(f"No ready local model route{detail}.")
    if len(matches) == 1:
        return matches[0]
    if not interactive:
        choices = ", ".join(route.selector for route in matches)
        raise click.ClickException(f"Ambiguous live route; choose one of: {choices}")
    for index, route in enumerate(matches, start=1):
        click.echo(f"  {index}) {route.label}", err=True)
    choice = click.prompt(
        "Select local route", type=click.IntRange(1, len(matches)), err=True
    )
    return matches[choice - 1]


def _explicit_live_route(
    url: str,
    model_id: str | None,
    api_key: str | None,
) -> LiveRoute:
    """Probe and validate an explicitly supplied local endpoint."""
    base_url = url.rstrip("/")
    if not base_url or any(ord(character) < 32 for character in base_url):
        raise click.ClickException("Explicit local endpoint URL is invalid.")
    probe = _probe_endpoint(base_url, api_key)
    if probe.readiness != "ready":
        raise click.ClickException(
            f"Explicit local endpoint is not ready: {probe.readiness}."
        )
    models = {model.model_id: model for model in probe.models}
    if model_id is not None and model_id not in models:
        raise click.ClickException(
            f"Model {model_id!r} is not reported by the explicit local endpoint."
        )
    if model_id is None and len(models) != 1:
        choices = ", ".join(sorted(models))
        raise click.ClickException(
            f"Explicit local endpoint reports several models; select one of: {choices}"
        )
    selected_id = model_id or next(iter(models))
    selected = models[selected_id]
    return LiveRoute(
        model_id=selected.model_id,
        node="explicit",
        port=0,
        gpu_count=0,
        job_id="explicit",
        base_url=base_url,
        max_context=selected.max_context,
        readiness=probe.readiness,
        job_name="explicit",
    )


# Display names for engine types (TOML stores the lowercase launcher key).
_ENGINE_LABELS = {"vllm": "vLLM", "sglang": "SGLang", "ktransformers": "KTransformers"}


def _fmt_uptime(slurm_time: str) -> str:
    """Render squeue ``%M`` (``[DD-]HH:MM:SS`` or ``MM:SS``) compactly.

    ``33:10`` → ``33m``, ``1:02:03`` → ``1h02m``, ``2-03:04:05`` → ``2d03h``.
    Falls back to the raw value on any unexpected shape.
    """
    raw = slurm_time.strip()
    days, _, hms = raw.partition("-")
    if not hms:
        hms, days = days, ""
    parts = hms.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return raw
    if days:
        return f"{int(days)}d{nums[0]:02d}h"
    if len(nums) == 3:
        return f"{nums[0]}h{nums[1]:02d}m"
    if len(nums) == 2:
        return f"{nums[0]}m"
    return raw


def _fmt_context(n: int) -> str:
    """Human-readable context length, e.g. 1048576 → ``1.0M``."""
    if n >= 1_000_000:
        return f"{n / 1_048_576:.1f}M"
    if n >= 1_000:
        return f"{n // 1024}K"
    return str(n)


def _engine_facts(profile) -> str:
    """One-line engine/model summary for the status detail block.

    Example: ``vLLM · TP=8 · ctx 1.0M · fp8 KV · MTP×5 · 640G``.
    """
    e = profile.engine
    engine_label = _ENGINE_LABELS.get(e.type, e.type)
    # Served context = max_total_tokens when set (the real cap), else the
    # model's theoretical max_context.
    served_ctx = e.max_total_tokens or profile.model.max_context
    parts = [
        engine_label,
        f"TP={e.tensor_parallel}",
        f"ctx {_fmt_context(served_ctx)}",
    ]
    if e.kv_cache_dtype:
        parts.append(f"{e.kv_cache_dtype} KV")
    if e.speculative_method:
        spec = e.speculative_method.upper()
        if e.speculative_num_tokens:
            spec += f"×{e.speculative_num_tokens}"
        parts.append(spec)
    parts.append(profile.slurm.memory)
    return " · ".join(parts)


def _update_dotenv_key(
    path: Path,
    key: str,
    value: str,
    *,
    header: str | None = None,
    mode: int = 0o600,
) -> None:
    """Set *key*=*value* in a dotenv file, preserving other lines.

    If the key exists, its line is replaced in-place.  If not, the
    key=value pair is appended.  Write is atomic (tmp + rename).
    """
    lines: list[str] = []
    replaced = False

    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _, _ = stripped.partition("=")
                if k.strip() == key:
                    lines.append(f"{key}={value}")
                    replaced = True
                    continue
            lines.append(line)

    if not replaced:
        if header and not lines:
            lines.append(header)
        lines.append(f"{key}={value}")

    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.chmod(mode)
    tmp.rename(path)


@agent.command(name="key")
@click.option("--reveal", is_flag=True, help="Show the full key in plaintext.")
@click.option(
    "--rotate", is_flag=True, help="Generate a new key, update configs, and restart."
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def key_command(reveal: bool, rotate: bool, yes: bool) -> None:
    """Show or rotate the API key for model serving.

    \b
    Without flags, shows the current key (masked).
    With --reveal, shows the full key.
    With --rotate, performs a full key rotation:
      1. Generates a new API key.
      2. Saves to agents/.env (mode 600, owner-only).
      3. Updates ~/.hermes/.env with OPENAI_API_KEY.
      4. Restarts the model server with the new key.

    Access control: the key file is mode 600 (owner read/write
    only).  Distribute the key to team members out-of-band;
    they store it in their own ~/.hermes/.env.
    """
    site = SiteConfig.from_env()
    key_path = site.api_key_file

    if not rotate:
        # Show mode
        try:
            token = _read_key_file(key_path)
        except PermissionError:
            raise click.ClickException(
                f"Permission denied: {key_path}\nOnly the key owner can read this file."
            ) from None

        if not token:
            console.print(f"No key configured at {key_path}")
            console.print("Generate one with: imas-ambix agent key --rotate")
            return

        console.print(f"File: {key_path}")
        console.print(f"Key:  {token if reveal else _mask_key(token)}")
        if not reveal:
            console.print("(use --reveal to show the full key)")
        return

    # Rotate mode
    import secrets
    import time as _time

    from imas_ambix.agent.slurm import generate_serve_script, submit_script

    existing = _read_key_file(key_path)

    if existing and not yes:
        console.print(f"[yellow]Current key:[/] {_mask_key(existing)}")
        if not click.confirm("Rotate to a new key?"):
            console.print("Aborted.")
            return

    token = secrets.token_urlsafe(32)

    # 1. Write to shared agents/.env (owner-only)
    _update_dotenv_key(
        key_path,
        "AMBIX_AGENT_API_KEY",
        token,
        header="# Ambix agent API key — managed by 'imas-ambix agent key'",
        mode=0o600,
    )
    action = "Rotated" if existing else "Generated"
    console.print(f"[green]{action} API key[/]")
    console.print(f"  Key:  {token}")
    console.print(f"  File: {key_path} (mode 600)")

    # 2. Update ~/.hermes/.env (owner-only)
    hermes_env = Path.home() / ".hermes" / ".env"
    if hermes_env.parent.is_dir():
        _update_dotenv_key(
            hermes_env,
            "OPENAI_API_KEY",
            token,
            mode=0o600,
        )
        console.print(f"  Updated: {hermes_env} (mode 600)")
    else:
        console.print(
            f"  [yellow]~/.hermes/ not found — set OPENAI_API_KEY={token} manually[/]"
        )

    # 3. Restart model server with new key
    console.print()
    default = _default_profile()
    if not default:
        console.print(
            "[yellow]No default profile set — restart the server manually:[/]"
        )
        console.print("  imas-ambix agent serve <profile>")
        return

    try:
        profile = load_profile(default)
    except FileNotFoundError:
        console.print(f"[yellow]Profile '{default}' not found — restart manually.[/]")
        return

    port = site.default_port
    user = os.environ.get("USER") or getpass.getuser()

    # Cancel active serve jobs
    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-A", site.account, "-o", "%i|%j", "-t", "R,PD"],
        capture_output=True,
        text=True,
        check=False,
    )
    targets: list[str] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        job_id, job_name = line.split("|", 1)
        if job_name.strip() == profile.slug:
            targets.append(job_id.strip())

    if targets:
        console.print(f"Cancelling {len(targets)} active job(s): " + ", ".join(targets))
        subprocess.run(
            ["scancel"] + targets,
            capture_output=True,
            text=True,
            check=False,
        )
        for _attempt in range(30):
            _time.sleep(2)
            check = subprocess.run(
                ["squeue", "-h", "-j", ",".join(targets), "-o", "%i"],
                capture_output=True,
                text=True,
                check=False,
            )
            if not check.stdout.strip():
                break
        console.print("[green]Old job(s) cancelled.[/]")

    # Submit new serve job with the new key
    script = generate_serve_script(profile, site, port=port, api_key=token)
    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"Submitted serve job {job_id} for {profile.slug} on port {port} "
        "(API key enabled)."
    )


@agent.command(name="clive")
@click.argument("slug", required=False, default=None)
@click.option(
    "--deploy",
    is_flag=True,
    help="Write the generated launcher to the shared GPFS path (default action).",
)
@click.option(
    "--mode",
    type=click.Choice(("local", "hybrid"), case_sensitive=True),
    default="local",
    show_default=True,
    help="Model scope; hybrid adds the hosted frontier slots.",
)
@click.option("--print", "print_only", is_flag=True, help="Print the script to stdout.")
@click.option(
    "--path",
    "show_path",
    is_flag=True,
    help="Show the deployed path and the PATH line to add to your shell.",
)
@click.option(
    "--destination",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    metavar="PATH",
    help="Write the launcher to PATH instead of the shared GPFS location.",
)
def clive_command(
    slug: str | None,
    deploy: bool,
    mode: str,
    print_only: bool,
    show_path: bool,
    destination: Path | None,
) -> None:
    """Generate and deploy the standalone global ``clive`` launcher.

    The distinct site-global origin is resolved from operator config at
    generation time. With no flags, deploys the launcher to
    ``{base_dir}/agents/``. A profile is consulted only when explicitly
    installing the optional OpenRouter proxy.
    """
    from imas_ambix.agent.clive import generate_clive_script

    site = SiteConfig.from_env()
    if slug is not None and mode != "hybrid":
        raise click.ClickException("A profile slug applies only with --mode hybrid.")
    proxy_native_release = None
    if mode == "hybrid":
        proxy_native_release = _load_profile(slug).model.served_name
    clive_script = generate_clive_script(
        site, mode=mode, openrouter_native_release=proxy_native_release
    )
    deploy_path = destination or site.clive_path

    if print_only:
        # Emit verbatim (no rich newline/markup), byte-identical to deploy.
        click.echo(clive_script, nl=False)
        return

    if show_path:
        console.print(f"clive:    {deploy_path}")
        console.print(f"Global endpoint: {site.global_origin}")
        console.print("Add to your ~/.bashrc to run as a bare command:")
        console.print(
            f"  [dim][[ -d {deploy_path.parent} ]] && "
            f'export PATH="{deploy_path.parent}:$PATH"[/]'
        )
        return

    # Default action: deploy the shared launcher from the repository to /work.
    _ = deploy  # deploy is the default; the flag is for explicitness only
    _deploy_launcher("clive", deploy_path, clive_script)

    if mode == "hybrid":
        assert proxy_native_release is not None
        _deploy_openrouter_proxy(site, proxy_native_release)

    console.print(f"  Global endpoint: {site.global_origin}")
    console.print("  PATH line: [cyan]imas-ambix agent clive --path[/]")


def _deploy_openrouter_proxy(site: SiteConfig, native_release: str) -> None:
    """Install the explicitly requested per-user proxy artifacts and unit."""
    from imas_ambix.agent.litellm_config import generate_litellm_config
    from imas_ambix.agent.litellm_service import (
        generate_litellm_env_helper,
        generate_litellm_service,
    )

    executable = Path.home() / ".local" / "bin" / "litellm"
    if not executable.is_file():
        raise click.ClickException(
            f"LiteLLM is absent at {executable}; "
            "use plain clive for the global release."
        )

    _deploy_launcher(
        "litellm_config.yaml",
        site.litellm_config_path,
        generate_litellm_config(site, native_release),
        0o644,
    )
    _deploy_launcher(
        "imas-ambix-llm-env.sh",
        site.litellm_env_helper_path,
        generate_litellm_env_helper(site),
    )

    service_path = site.litellm_service_path
    try:
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(generate_litellm_service(site), encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    except OSError as exc:
        raise click.ClickException(
            f"Could not install the OpenRouter proxy unit: {exc}. "
            "Use plain clive for the global release."
        ) from exc
    console.print(f"[green]Installed[/] {service_path.name} (per-user systemd unit)")


def _deploy_launcher(name: str, path, content: str, mode: int = 0o755) -> None:
    """Write a generated artifact to *path* (group-matched), or raise."""
    import shutil

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(mode)
        # Match the parent dir's group so the storage group can use it.
        shutil.chown(path, group=path.parent.stat().st_gid)
    except OSError as exc:
        raise click.ClickException(f"Failed to deploy {name} to {path}: {exc}") from exc
    console.print(f"[green]Deployed {name}[/] → {path}")


@agent.command()
@click.argument("slug", required=False, default=None)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SLURM script instead of submitting it.",
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=None,
    help="Port to expose on the compute node.",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="API key for authenticating client requests (also AMBIX_AGENT_API_KEY).",
)
@click.option(
    "--auth",
    "auth",
    is_flag=True,
    help="Require an API key on /v1/* requests, resolved from --api-key, "
    "AMBIX_AGENT_API_KEY, or the shared key file. The default is an open "
    "endpoint: the cluster is already the authentication boundary, and a key "
    "readable only inside one storage group would lock consumers out of it.",
)
@click.option(
    "--gpus",
    type=int,
    default=None,
    help="Override number of GPUs (and tensor-parallel size). "
    "Scales cpus and memory proportionally from the profile default.",
)
@click.option(
    "--cpus",
    type=click.IntRange(1, 64),
    default=None,
    help="Override host cores, applied after any --gpus scaling. This node "
    "carries two overlapping 30-core reservations, so cores are the scarce "
    "resource a serve competes for: an inference server spends most of its "
    "time blocked on the device, and the proportional default reserves cores "
    "a co-running job could use. Probe with `srun --test-only` before "
    "committing to a value.",
)
def restart(
    slug: str | None,
    dry_run: bool,
    port: int | None,
    api_key: str | None,
    auth: bool,
    gpus: int | None,
    cpus: int | None,
) -> None:
    """Restart a model serving job (shutdown + serve).

    Cancels any active serve job for the profile, waits for it to
    finish, then submits a new serve job.  If no active job exists,
    starts a fresh one.
    """
    import time as _time

    from imas_ambix.agent.slurm import generate_serve_script, submit_script

    profile = _load_profile(slug)
    if gpus is not None:
        profile = _scale_profile(profile, gpus)
    if cpus is not None:
        profile = profile.model_copy(
            update={"slurm": profile.slurm.model_copy(update={"cpus": cpus})}
        )
    site = SiteConfig.from_env()
    resolved_port = port if port is not None else site.default_port
    resolved_key = _resolve_serve_auth(auth, api_key, site)

    # Find active serve jobs for this profile
    user = os.environ.get("USER") or getpass.getuser()
    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-A", site.account, "-o", "%i|%j", "-t", "R,PD"],
        capture_output=True,
        text=True,
        check=False,
    )
    targets: list[str] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        job_id, job_name = line.split("|", 1)
        if job_name.strip() == profile.slug:
            targets.append(job_id.strip())

    if targets:
        console.print(f"Cancelling {len(targets)} active job(s): " + ", ".join(targets))
        cancel = subprocess.run(
            ["scancel"] + targets,
            capture_output=True,
            text=True,
            check=False,
        )
        if cancel.returncode != 0:
            raise click.ClickException(cancel.stderr.strip() or "scancel failed")

        # Wait for jobs to drain
        for _attempt in range(30):
            _time.sleep(2)
            check = subprocess.run(
                ["squeue", "-h", "-j", ",".join(targets), "-o", "%i"],
                capture_output=True,
                text=True,
                check=False,
            )
            if not check.stdout.strip():
                break
        else:
            raise click.ClickException(
                "Timed out waiting for old job(s) to stop (60s). Check squeue manually."
            )
        console.print("[green]Old job(s) cancelled.[/]")
    else:
        console.print("No active serve job found — starting fresh.")

    # Submit new serve job
    script = generate_serve_script(
        profile, site, port=resolved_port, api_key=resolved_key
    )
    if dry_run:
        if resolved_key:
            script = script.replace(resolved_key, "****")
        console.print(script, markup=False, highlight=False, soft_wrap=True)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    key_note = " (API key enabled)" if resolved_key else " (NO AUTH — open endpoint)"
    gpu_note = (
        f" ({profile.slurm.gpus}×GPU, {profile.slurm.cpus} CPU)"
        if gpus is not None or cpus is not None
        else ""
    )
    message = (
        f"Submitted serve job {job_id} for {profile.slug}{gpu_note} "
        f"on port {resolved_port}{key_note}."
    )
    console.print(message)


@agent.command()
@click.argument("slug", required=False)
@click.option(
    "--url", default=None, help="Server base URL (default: from site config)."
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Model name for API requests.",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for authenticated endpoints (or set AMBIX_AGENT_API_KEY).",
)
@click.option(
    "--category",
    multiple=True,
    type=click.Choice(
        ["throughput", "prefill", "context", "tools", "reasoning", "concurrency"]
    ),
    help="Categories to run (default: all).",
)
@click.option("--repeat", type=int, default=1, help="Repeat each test N times.")
@click.option(
    "--max-context", type=int, default=None, help="Max context for context tests."
)
@click.option("--json", "json_output", is_flag=True, help="Output raw JSON only.")
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Write JSON results to file (default: ~/.local/share/ambix/bench/).",
)
@click.option("--no-save", is_flag=True, help="Disable auto-saving results.")
@click.option("--warmup/--no-warmup", default=True, help="Run warmup request.")
def bench(
    slug: str | None,
    url: str | None,
    model_name: str | None,
    api_key: str | None,
    category: tuple[str, ...],
    repeat: int,
    max_context: int | None,
    json_output: bool,
    output_path: str | None,
    no_save: bool,
    warmup: bool,
) -> None:
    """Comprehensive LLM benchmark suite.

    \b
    Run with defaults (uses default profile + site GPU host):
        imas-ambix agent bench

    \b
    Run against any endpoint:
        imas-ambix agent bench --url http://host:18800 --model my-model

    Results are auto-saved to ~/.local/share/ambix/bench/ unless --no-save.
    """
    import json as json_mod
    import urllib.error
    import urllib.request

    from imas_ambix.agent.bench import _auth_headers, run_benchmark

    resolved_key = _resolve_api_key(api_key)

    # Resolve base_url and model — try slug, then default profile, then url-only
    resolved_slug = slug or _default_profile()
    serve_job: str | None = None
    if resolved_slug:
        profile = _load_profile(resolved_slug)
        # Attribute the run to the deployment that is actually serving, not to
        # the profile's default card count.
        serve_gpus, serve_job = _running_serve_gpus(
            resolved_slug, SiteConfig.from_env()
        )
        if serve_gpus:
            # Resolve exactly as the serve path does. for_gpus alone only
            # applies a DECLARED variant, so a card count reached by
            # proportional scaling kept the profile's default width and the
            # run was filed under the wrong topology.
            profile = _scale_profile(profile, serve_gpus)
        base_url = url or _default_url() or "http://localhost:18800"
        model = model_name or profile.model.served_name
    elif url:
        profile = None
        base_url = url
        model = model_name or "default"
    else:
        raise click.ClickException(
            "Provide a profile slug or --url. "
            "Example: imas-ambix agent bench deepseek-v4-flash"
        )

    # Health check with auth
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/models", headers=_auth_headers(resolved_key)
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            models_data = json_mod.loads(resp.read())
            available = [m["id"] for m in models_data.get("data", [])]
            if model not in available and available:
                console.print(
                    f"[yellow]Warning:[/] '{model}' not in server models: {available}"
                )
                model = available[0]
                console.print(f"Using '{model}' instead.")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise click.ClickException(f"Cannot reach server at {base_url}: {exc}") from exc

    cats = list(category) if category else None

    console.print(f"\n[bold]Benchmarking[/] {model} at {base_url}")
    if cats:
        console.print(f"Categories: {', '.join(cats)}")
    console.print()

    report: BenchReport = run_benchmark(
        base_url,
        model,
        categories=cats,
        repeat=repeat,
        max_context=max_context,
        warmup=warmup,
        api_key=resolved_key,
        # Without this the saved run records no serving configuration, and an
        # unattributable run cannot be compared against another.
        profile=profile,
        serve_job_id=serve_job,
    )

    # Auto-save results
    save_path: Path | None = None
    if output_path:
        save_path = Path(output_path)
    elif not no_save:
        import datetime as _dt

        bench_dir = Path.home() / ".local" / "share" / "ambix" / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = bench_dir / f"{model}_{ts}.json"

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report.to_json(), encoding="utf-8")

    # JSON-only output
    if json_output:
        console.print(report.to_json(), markup=False, highlight=False, soft_wrap=True)
        if save_path:
            console.print(f"[dim]Results saved to {save_path}[/]", err=True)
        return

    # Rich per-category tables
    _render_report(report, model, repeat)

    if save_path:
        console.print(f"\n[dim]Results saved to {save_path}[/]")


def _render_report(report: BenchReport, model: str, repeat: int = 1) -> None:
    """Render benchmark results as rich tables to the console."""
    from rich.table import Table as RichTable

    grouped: dict[str, list] = {}
    for r in report.results:
        grouped.setdefault(r.category, []).append(r)

    for cat, results in grouped.items():
        if cat == "concurrency":
            _render_concurrency(results, repeat)
            continue

        table = RichTable(title=f"{cat.title()}")
        table.add_column("Test", style="cyan")
        table.add_column("Status")
        table.add_column("Prompt", justify="right")
        table.add_column("Comp", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Decode TPS", justify="right", style="bold green")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Notes")

        for r in results:
            status_str = _status_icon(r.status)
            notes = r.error or r.metadata.get("note", "")
            tps = f"{r.decode_tps:.1f}" if r.decode_tps > 0 else "—"
            ttft = r.time_to_first_token_s
            ttft_s = f"{ttft * 1000:.0f}" if ttft > 0 else "—"
            note_s = (notes[:60] + "…") if len(notes) > 60 else notes
            rep_tag = f" #{r.repeat_index}" if repeat > 1 else ""
            table.add_row(
                f"{r.test_name}{rep_tag}",
                status_str,
                str(r.prompt_tokens),
                str(r.completion_tokens),
                f"{r.total_time_s:.2f}",
                tps,
                ttft_s,
                note_s,
            )
        console.print(table)
        console.print()

    # Summary
    summary = report.summary()
    if summary:
        stbl = RichTable(title=f"Summary — {model}")
        stbl.add_column("Category", style="cyan")
        stbl.add_column("Passed", justify="right")
        stbl.add_column("Failed", justify="right")
        stbl.add_column("Skipped", justify="right")
        stbl.add_column("Avg TPS", justify="right", style="bold green")
        stbl.add_column("Avg TTFT (ms)", justify="right")
        stbl.add_column("p95 TTFT (ms)", justify="right")

        for cat, stats in summary.items():
            p95 = report.percentiles(cat, "time_to_first_token_s")
            stbl.add_row(
                cat,
                str(stats["passed"]),
                str(stats["failed"]),
                str(stats["skipped"]),
                str(stats["avg_decode_tps"]),
                str(stats["avg_ttft_ms"]),
                f"{p95['p95'] * 1000:.0f}" if p95["p95"] > 0 else "—",
            )
        console.print(stbl)


def _status_icon(status: str) -> str:
    return {
        "passed": "[green]✓[/]",
        "failed": "[red]✗[/]",
        "skipped": "[yellow]⊘[/]",
        "error": "[red]E[/]",
    }.get(status, status)


def _render_concurrency(results: list, repeat: int) -> None:
    """Render concurrency results with aggregate TPS scaling."""
    from rich.table import Table as RichTable

    # Group by concurrency level
    levels: dict[str, list] = {}
    for r in results:
        levels.setdefault(r.test_name, []).append(r)

    # Per-worker detail table
    table = RichTable(title="Concurrency")
    table.add_column("Workers", style="cyan", justify="right")
    table.add_column("Status")
    table.add_column("Per-Stream TPS", justify="right", style="bold green")
    table.add_column("Aggregate TPS", justify="right", style="bold magenta")
    table.add_column("Wall Time (s)", justify="right")
    table.add_column("TTFT (ms)", justify="right")

    for _test_name, level_results in levels.items():
        n_workers = level_results[0].metadata.get("n_workers", "?")
        ok = [r for r in level_results if r.ok]
        tps_vals = [r.decode_tps for r in ok if r.decode_tps > 0]
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
        agg_tps = level_results[0].metadata.get("aggregate_tps", 0)
        wall = level_results[0].metadata.get("wall_time", 0)
        ttft_vals = [r.time_to_first_token_s for r in ok if r.time_to_first_token_s > 0]
        avg_ttft = sum(ttft_vals) / len(ttft_vals) * 1000 if ttft_vals else 0
        all_ok = all(r.ok for r in level_results)
        table.add_row(
            str(n_workers),
            _status_icon("passed" if all_ok else "failed"),
            f"{avg_tps:.1f}",
            f"{agg_tps:.1f}",
            f"{wall:.2f}",
            f"{avg_ttft:.0f}",
        )

    console.print(table)
    console.print()


@agent.command(name="bench-compare")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--model",
    default=None,
    help="Restrict discovery to saved runs whose model name contains this text.",
)
@click.option(
    "--last",
    type=int,
    default=None,
    help="Compare the N most recent saved runs (default 2 when no PATHS given).",
)
@click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory of saved runs (default: ~/.local/share/ambix/bench).",
)
def bench_compare(
    paths: tuple[str, ...],
    model: str | None,
    last: int | None,
    directory: str | None,
) -> None:
    """Compare saved benchmark runs, or summarise a single run.

    \b
    Two most recent runs:              imas-ambix agent bench-compare
    Two most recent for one model:     imas-ambix agent bench-compare --model glm
    Specific runs:                     imas-ambix agent bench-compare a.json b.json

    A run only compares meaningfully when it carries provenance (serving
    configuration and engine version); runs saved before provenance capture
    show their configuration as unknown rather than a guessed value.
    """
    from imas_ambix.agent.bench_report import (
        BenchReportError,
        compare_runs,
        discover_reports,
        load_report,
        render_comparison,
        render_run,
    )

    try:
        if paths:
            selected = [Path(p) for p in paths]
        else:
            selected = discover_reports(
                directory=directory, model=model, limit=last or 2
            )
        if not selected:
            raise click.ClickException(
                "No saved benchmark runs found. Run 'imas-ambix agent bench' first."
            )
        reports = [load_report(p) for p in selected]
    except BenchReportError as exc:
        raise click.ClickException(str(exc)) from exc

    for path in selected:
        console.print(f"[dim]{path}[/]")

    if len(reports) == 1:
        render_run(reports[0])
        return
    render_comparison(compare_runs(reports))


def _running_serve_gpus(slug: str, site: SiteConfig) -> tuple[int | None, str | None]:
    """Return ``(gpu_count, job_id)`` for *slug*'s RUNNING serve job.

    A profile describes a default deployment, but a card count is chosen at
    launch, so the profile alone cannot say which variant is answering. Ask the
    scheduler what is actually running instead of assuming the default, or a
    benchmark of one deployment gets attributed to another.
    """
    user = os.environ.get("USER") or getpass.getuser()
    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-A", site.account, "-o", "%i|%j|%T|%b"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None, None
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4 or parts[1].strip() != slug or parts[2].strip() != "RUNNING":
            continue
        # %b renders the generic-resource request, e.g. "gres/gpu:4".
        tail = parts[3].strip().rsplit(":", maxsplit=1)[-1]
        return (int(tail) if tail.isdigit() else None), parts[0].strip()
    return None, None


def _engine_pyproject(engine: str) -> str:
    """Return the bundled pyproject.toml content for *engine*."""
    from importlib import resources

    pkg = resources.files("imas_ambix.agent.envs") / engine / "pyproject.toml"
    return pkg.read_text(encoding="utf-8")


def _metadata_version_command(package: str, label: str, *, ok: bool = False) -> str:
    """Return a shell command that reports installed package metadata."""
    values = f"{label!r}, m.version({package!r})"
    if ok:
        values += ", 'OK'"
    code = f"import importlib.metadata as m; print({values})"
    return f'"$PYTHON" -c {shlex.quote(code)}'


def _engine_runtime_check_script(
    engine: str,
    site: SiteConfig,
    dependency_job_id: str,
    expected_identity: str,
) -> str:
    """Return a serving-node check for a completed network install.

    The engine environment is written from a network-enabled compute node but
    consumed on the GPU node.  Checking it in the producer allocation cannot
    establish that the serving mount resolves the same directory entries, so
    this dependent job verifies the consumer-visible path after the producer
    has exited.
    """
    python = site.python_path(engine)
    python_q = shlex.quote(str(python))
    identity_file_q = shlex.quote(str(site.env_dir(engine) / ".ambix-setup-identity"))
    expected_identity_q = shlex.quote(expected_identity)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=ambix-runtime-check-{engine}",
        f"#SBATCH --partition={site.partition}",
    ]
    if site.reservation:
        lines.append(f"#SBATCH --reservation={site.reservation}")
    lines += [
        f"#SBATCH --account={site.account}",
        f"#SBATCH --dependency=afterok:{dependency_job_id}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=1G",
        "#SBATCH --time=00:10:00",
        f"#SBATCH --output=ambix-runtime-check-{engine}-%j.log",
        "",
        "set -euo pipefail",
        "export TMPDIR=/tmp",
        "",
        f"PYTHON={python_q}",
        f"IDENTITY_FILE={identity_file_q}",
        f"EXPECTED_SETUP_IDENTITY={expected_identity_q}",
        'if [ ! -x "$PYTHON" ]; then',
        '    echo "ERROR: runtime node cannot execute $PYTHON" >&2',
        '    echo "The network install is not durable on the serving filesystem." >&2',
        "    exit 1",
        "fi",
        "",
        'if [ ! -r "$IDENTITY_FILE" ]; then',
        '    echo "ERROR: runtime node cannot read $IDENTITY_FILE" >&2',
        "    exit 1",
        "fi",
        'ACTUAL_SETUP_IDENTITY=$(cat "$IDENTITY_FILE")',
        'if [ "$ACTUAL_SETUP_IDENTITY" != "$EXPECTED_SETUP_IDENTITY" ]; then',
        (
            '    echo "ERROR: serving environment does not match '
            'the completed network install" >&2'
        ),
        "    exit 1",
        "fi",
        "",
        'stat -Lc "interpreter=%n device=%d inode=%i size=%s" "$PYTHON"',
        '"$PYTHON" --version',
    ]

    if engine == "vllm":
        lines += [
            _metadata_version_command("vllm", "vLLM", ok=True),
            _metadata_version_command("torch", "PyTorch"),
            _metadata_version_command("transformers", "transformers"),
        ]
    elif engine == "sglang":
        lines += [
            _metadata_version_command("sglang", "SGLang", ok=True),
            _metadata_version_command("torch", "PyTorch"),
        ]

    lines += ["", 'echo "=== Runtime verification complete ==="', ""]
    return "\n".join(lines)


@agent.command()
@click.argument("engine", type=click.Choice(ENGINE_TYPES))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SLURM setup script instead of submitting it.",
)
def setup(engine: str, dry_run: bool) -> None:
    """Create or sync the uv-managed venv for an engine.

    Copies the engine's pyproject.toml to its per-user environment directory
    and runs ``uv sync`` via a SLURM job on a network-enabled partition.
    """
    site = SiteConfig.from_env()
    env_dir = site.env_dir(engine)
    pyproject_content = _engine_pyproject(engine)
    setup_identity = secrets.token_hex(16)

    from imas_ambix.agent.slurm import submit_script

    env_dir_q = shlex.quote(str(env_dir))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=ambix-setup-{engine}",
        f"#SBATCH --partition={site.download_partition}",
        f"#SBATCH --account={site.account}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=16G",
        "#SBATCH --time=01:00:00",
        f"#SBATCH --output=ambix-setup-{engine}-%j.log",
        "",
        "set -euo pipefail",
        "",
        "# uv's hardlink mode silently fails on cross-filesystem GPFS mounts,",
        "# leaving nvidia lib directories empty. Force full copies.",
        "export UV_LINK_MODE=copy",
        "",
        f"ENV_DIR={env_dir_q}",
        'mkdir -p "$ENV_DIR"',
        'cd "$ENV_DIR"',
        "",
        f"SETUP_IDENTITY={setup_identity}",
        f"MIN_FREE_KIB=$(({site.engine_env_min_free_gb} * 1024 * 1024))",
        "AVAILABLE_KIB=$(df -Pk \"$ENV_DIR\" | awk 'NR == 2 {print $4}')",
        'case "$AVAILABLE_KIB" in',
        (
            '    ""|*[!0-9]*) echo "ERROR: could not measure free space '
            'for $ENV_DIR" >&2; exit 1 ;;'
        ),
        "esac",
        'if [ "$AVAILABLE_KIB" -lt "$MIN_FREE_KIB" ]; then',
        (
            '    echo "ERROR: $ENV_DIR has $AVAILABLE_KIB KiB free; '
            '$MIN_FREE_KIB KiB required" >&2'
        ),
        "    exit 1",
        "fi",
        'echo "Capacity preflight: $AVAILABLE_KIB KiB free"',
        "",
        "# Write pyproject.toml from bundled package data",
        "cat > pyproject.toml << 'PYPROJECT_EOF'",
        pyproject_content.rstrip(),
        "PYPROJECT_EOF",
        "",
        "# Ensure uv is available",
        "if ! command -v uv &>/dev/null; then",
        (
            '    echo "ERROR: uv not found. Install with: '
            'curl -LsSf https://astral.sh/uv/install.sh | sh"'
        ),
        "    exit 1",
        "fi",
        "",
        f'echo "=== Setting up {engine} environment in $ENV_DIR ==="',
        "",
        "# Remove stale lockfile so uv regenerates it from current pyproject.toml",
        "rm -f uv.lock",
        "",
    ]

    # vLLM ships a linux x86_64 wheel whose manylinux tag may exceed SDCC's
    # glibc 2.34. vLLM 0.23.0+ targets manylinux_2_28 (glibc 2.28 ≤ 2.34 — runs
    # natively, no rename); older wheels (≤0.20.x) targeted manylinux_2_35
    # (glibc 2.35 > 2.34 — needs the tag renamed down to 2_17 to install).
    # Wheelhouse cache key is the resolved vLLM version, so a version bump in
    # pyproject re-downloads instead of reusing a stale wheel.
    if engine == "vllm":
        lines += [
            "mkdir -p wheelhouse",
            "# Resolve the latest vLLM linux x86_64 wheel via the PyPI JSON API",
            "# (URL, filename, and the manylinux glibc minor it targets).",
            '    read VLLM_URL VLLM_WHEEL VLLM_GLIBC < <(python3 -c "',
            "import urllib.request, json, re",
            "resp = urllib.request.urlopen('https://pypi.org/pypi/vllm/json')",
            "data = json.loads(resp.read())",
            "for u in data['urls']:",
            "    fn = u['filename']",
            "    m = re.search(r'manylinux_2_(\\d+)_x86_64', fn)",
            "    if m and fn.endswith('.whl'):",
            "        print(u['url'], fn, m.group(1)); break",
            '")',
            '    if [ -z "${VLLM_WHEEL:-}" ]; then',
            (
                '        echo "ERROR: could not resolve a vLLM linux x86_64 '
                'wheel from PyPI"; exit 1'
            ),
            "    fi",
            "    # Wheel that uv will install — renamed to manylinux_2_17 only when",
            "    # the source tag (glibc minor) is newer than SDCC's glibc 2.34.",
            '    if [ "$VLLM_GLIBC" -gt 34 ]; then',
            (
                '        WHEEL_FINAL=$(echo "$VLLM_WHEEL" | '
                'sed "s/manylinux_2_${VLLM_GLIBC}/manylinux_2_17/")'
            ),
            "    else",
            '        WHEEL_FINAL="$VLLM_WHEEL"',
            "    fi",
            '    if [ ! -f "wheelhouse/$WHEEL_FINAL" ]; then',
            (
                '        echo "Downloading vLLM wheel $VLLM_WHEEL '
                '(glibc 2.$VLLM_GLIBC target)..."'
            ),
            '        curl -fSL "$VLLM_URL" -o "wheelhouse/$VLLM_WHEEL"',
            '        if [ "$WHEEL_FINAL" != "$VLLM_WHEEL" ]; then',
            '            mv "wheelhouse/$VLLM_WHEEL" "wheelhouse/$WHEEL_FINAL"',
            '            echo "Renamed manylinux tag → $WHEEL_FINAL"',
            "        fi",
            "    else",
            '        echo "Reusing cached wheelhouse/$WHEEL_FINAL"',
            "    fi",
            "    # Drop other cached vLLM wheels so the install glob is unambiguous.",
            (
                '    find wheelhouse -maxdepth 1 -name "vllm-*.whl" '
                '! -name "$WHEEL_FINAL" -delete'
            ),
            "",
        ]

    lines += [
        "# uv sync creates/updates .venv and installs all dependencies",
        "uv sync --python 3.12 -v 2>&1 | tail -50",
    ]

    # vLLM: install the renamed wheel into the synced venv
    if engine == "vllm":
        lines += [
            "",
            'echo "Installing vLLM from local wheelhouse..."',
            "# --no-deps: all dependencies already installed by uv sync above",
            (
                "uv pip install --no-deps --python .venv/bin/python "
                "wheelhouse/vllm-*x86_64.whl"
            ),
        ]

    lines += [
        "",
        'echo ""',
        'echo "=== Verifying installation ==="',
        'PYTHON="$ENV_DIR/.venv/bin/python"',
        '"$PYTHON" --version',
    ]

    # Engine-specific smoke tests (version check via metadata — no CUDA needed)
    if engine == "vllm":
        lines += [
            _metadata_version_command("vllm", "vLLM", ok=True),
            _metadata_version_command("torch", "PyTorch"),
            _metadata_version_command("transformers", "transformers"),
        ]
    elif engine == "sglang":
        lines += [
            _metadata_version_command("sglang", "SGLang", ok=True),
            _metadata_version_command("torch", "PyTorch"),
        ]

    lines.append("")
    lines.append('IDENTITY_FILE="$ENV_DIR/.ambix-setup-identity"')
    lines.append('printf \'%s\\n\' "$SETUP_IDENTITY" > "$IDENTITY_FILE"')
    lines.append('sync "$IDENTITY_FILE"')
    lines.append(
        "ENV_SIZE_BYTES=$(du -s --block-size=1 \"$ENV_DIR\" | awk '{print $1}')"
    )
    lines.append('case "$ENV_SIZE_BYTES" in')
    lines.append(
        '    ""|*[!0-9]*|0) echo "ERROR: could not measure installed '
        'environment" >&2; exit 1 ;;'
    )
    lines.append("esac")
    lines.append('echo "Installed environment size: $ENV_SIZE_BYTES bytes"')
    lines.append("")
    lines.append('echo "=== Network installation complete ==="')
    lines.append('echo "The dependent serving-node verification must pass before use."')
    lines.append("")

    script = "\n".join(lines)

    if dry_run:
        console.print(script, markup=False, highlight=False, soft_wrap=True)
        console.print(
            _engine_runtime_check_script(engine, site, "SETUP_JOB_ID", setup_identity),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    runtime_check_script = _engine_runtime_check_script(
        engine, site, job_id, setup_identity
    )
    try:
        runtime_check_job_id = submit_script(runtime_check_script)
    except RuntimeError as exc:
        raise click.ClickException(
            f"Network install job {job_id} was submitted, but its serving-node "
            f"verification could not be scheduled: {exc}. The environment is not ready."
        ) from exc
    console.print(
        f"Submitted network install job [bold]{job_id}[/] for [cyan]{engine}[/]."
    )
    console.print(
        f"Submitted dependent runtime verification job [bold]{runtime_check_job_id}[/]."
    )
    console.print(
        f"Environment is not ready until runtime verification job "
        f"{runtime_check_job_id} passes."
    )
    console.print(f"  Environment: {env_dir}")
    console.print(f"  Monitor: squeue -j {job_id},{runtime_check_job_id}")
    console.print(
        f"  Logs: ambix-setup-{engine}-{job_id}.log, "
        f"ambix-runtime-check-{engine}-{runtime_check_job_id}.log"
    )
