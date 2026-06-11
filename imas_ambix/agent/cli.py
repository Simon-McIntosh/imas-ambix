"""CLI commands for managing Ambix agent deployments."""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imas_ambix.agent.profile import SiteConfig, list_profiles, load_profile

console = Console()

# Engine types that have uv-managed environments
ENGINE_TYPES = ("vllm", "sglang")


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

    Adjusts ``engine.tensor_parallel``, ``slurm.gpus``, ``slurm.cpus``, and
    ``slurm.memory`` proportionally.  The caller's profile is not mutated.
    """
    base_gpus = profile.slurm.gpus
    if gpus == base_gpus:
        return profile
    ratio = gpus / base_gpus
    mem_str = profile.slurm.memory
    mem_val = int(mem_str[:-1])
    mem_unit = mem_str[-1]
    new_memory = f"{max(1, round(mem_val * ratio))}{mem_unit}"
    new_cpus = max(1, round(profile.slurm.cpus * ratio))
    return profile.model_copy(
        deep=True,
        update={
            "slurm": profile.slurm.model_copy(
                update={"gpus": gpus, "cpus": new_cpus, "memory": new_memory}
            ),
            "engine": profile.engine.model_copy(update={"tensor_parallel": gpus}),
        },
    )


@click.group()
def agent() -> None:
    """Manage LLM agent deployments on SLURM GPU clusters."""


@agent.command(name="list")
def list_command() -> None:
    """List available model profiles."""
    table = Table(title="Ambix agent profiles")
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Engine")
    table.add_column("Size")

    for slug in list_profiles():
        profile = _load_profile(slug)
        table.add_row(
            profile.slug,
            profile.model.name,
            profile.engine.type,
            f"{profile.model.size_gb} GB",
        )

    console.print(table)


@agent.command()
@click.argument("slug", required=False, default=None)
def info(slug: str | None) -> None:
    """Show detailed information for a model profile."""
    profile = _load_profile(slug)
    site = SiteConfig.from_env()

    table = Table(title=f"Profile: {profile.slug}", show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Model name", profile.model.name)
    table.add_row("HF repo", profile.model.hf_repo)
    table.add_row("Served name", profile.model.served_name)
    table.add_row("Engine", profile.engine.type)
    table.add_row("Tensor parallel", str(profile.engine.tensor_parallel))
    table.add_row("Attention backend", profile.engine.attention_backend)
    table.add_row("Max context", str(profile.model.max_context))
    table.add_row("Model size", f"{profile.model.size_gb} GB")
    table.add_row("SLURM GPUs", str(profile.slurm.gpus))
    table.add_row("SLURM CPUs", str(profile.slurm.cpus))
    table.add_row("SLURM memory", profile.slurm.memory)
    table.add_row("Serve time", profile.slurm.time_serve)
    table.add_row("Download time", profile.slurm.time_download)
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
    console.print(table)


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
        click.echo(script)
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
    "--gpus",
    type=int,
    default=None,
    help="Override number of GPUs (and tensor-parallel size). "
    "Scales cpus and memory proportionally from the profile default.",
)
def serve(
    slug: str | None,
    dry_run: bool,
    port: int | None,
    api_key: str | None,
    gpus: int | None,
) -> None:
    """Generate and submit a model serving job."""
    from imas_ambix.agent.slurm import generate_serve_script, submit_script

    profile = _load_profile(slug)
    if gpus is not None:
        profile = _scale_profile(profile, gpus)
    site = SiteConfig.from_env()
    resolved_port = port if port is not None else site.default_port
    resolved_key = _resolve_api_key(api_key)
    script = generate_serve_script(
        profile, site, port=resolved_port, api_key=resolved_key
    )

    if dry_run:
        if resolved_key:
            script = script.replace(resolved_key, "****")
        click.echo(script)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    key_note = " (API key enabled)" if resolved_key else ""
    gpu_note = f" ({profile.slurm.gpus}×GPU)" if gpus is not None else ""
    console.print(
        f"Submitted serve job {job_id} for {profile.slug}{gpu_note} on port {resolved_port}{key_note}."
    )


@agent.command()
def status() -> None:
    """Show active Ambix agent SLURM jobs."""
    from imas_ambix.agent.profile import SiteConfig

    user = os.environ.get("USER") or getpass.getuser()
    site = SiteConfig.from_env()
    command = (
        "squeue -u "
        f"{shlex.quote(user)} "
        f"-A {shlex.quote(site.account)} "
        '-o "%.10i %.20j %.8T %.10M %.6D %R" || true'
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "Failed to query squeue"
        raise click.ClickException(message)

    output = result.stdout.strip()
    if not output:
        console.print("No ambix agent jobs found.")
        return

    console.print(output)


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
        console.print("No ambix agent jobs found.")
        return

    # Filter by scope
    if cancel_all:
        targets = jobs
    else:
        resolved = _resolve_slug(slug) if slug else _default_profile()
        if resolved:
            targets = [(jid, jn) for jid, jn in jobs if jn == resolved]
        else:
            # No slug and no default — cancel serve jobs (match known profile slugs only)
            known = set(list_profiles())
            targets = [(jid, jn) for jid, jn in jobs if jn in known]

    if not targets:
        console.print("No matching jobs to cancel.")
        return

    # Show what will be cancelled
    console.print("[bold]Jobs to cancel:[/]")
    for job_id, job_name in targets:
        console.print(f"  {job_id}  {job_name}")

    if not yes:
        if not click.confirm("Proceed?"):
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
            console.print("Generate one with: ambix agent key --rotate")
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
        header="# Ambix agent API key — managed by 'ambix agent key'",
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
        console.print("  ambix agent serve <profile>")
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
        f"Submitted serve job {job_id} for {profile.slug} on port {port} (API key enabled)."
    )


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
    "--gpus",
    type=int,
    default=None,
    help="Override number of GPUs (and tensor-parallel size). "
    "Scales cpus and memory proportionally from the profile default.",
)
def restart(
    slug: str | None,
    dry_run: bool,
    port: int | None,
    api_key: str | None,
    gpus: int | None,
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
    site = SiteConfig.from_env()
    resolved_port = port if port is not None else site.default_port
    resolved_key = _resolve_api_key(api_key)

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
        click.echo(script)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    key_note = " (API key enabled)" if resolved_key else ""
    gpu_note = f" ({profile.slurm.gpus}×GPU)" if gpus is not None else ""
    console.print(
        f"Submitted serve job {job_id} for {profile.slug}{gpu_note} on port {resolved_port}{key_note}."
    )


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
        ambix agent bench

    \b
    Run against any endpoint:
        ambix agent bench --url http://host:18800 --model my-model

    Results are auto-saved to ~/.local/share/ambix/bench/ unless --no-save.
    """
    import json as json_mod
    import urllib.error
    import urllib.request

    from imas_ambix.agent.bench import BenchReport, _auth_headers, run_benchmark

    resolved_key = _resolve_api_key(api_key)

    # Resolve base_url and model — try slug, then default profile, then url-only
    resolved_slug = slug or _default_profile()
    if resolved_slug:
        profile = _load_profile(resolved_slug)
        base_url = url or _default_url() or "http://localhost:18800"
        model = model_name or profile.model.served_name
    elif url:
        base_url = url
        model = model_name or "default"
    else:
        raise click.ClickException(
            "Provide a profile slug or --url. "
            "Example: ambix agent bench deepseek-v4-flash"
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
        click.echo(report.to_json())
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

    for test_name, level_results in levels.items():
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


def _engine_pyproject(engine: str) -> str:
    """Return the bundled pyproject.toml content for *engine*."""
    from importlib import resources

    pkg = resources.files("imas_ambix.agent.envs") / engine / "pyproject.toml"
    return pkg.read_text(encoding="utf-8")


@agent.command()
@click.argument("engine", type=click.Choice(ENGINE_TYPES))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SLURM setup script instead of submitting it.",
)
def setup(engine: str, dry_run: bool) -> None:
    """Create or sync the uv-managed venv for an engine.

    Copies the engine's pyproject.toml to the workspace directory
    (e.g. /work/projects/imas_gpu/agents/vllm/) and runs ``uv sync``
    via a SLURM job on a network-enabled partition.
    """
    site = SiteConfig.from_env()
    env_dir = site.env_dir(engine)
    pyproject_content = _engine_pyproject(engine)

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
        "# Write pyproject.toml from bundled package data",
        "cat > pyproject.toml << 'PYPROJECT_EOF'",
        pyproject_content.rstrip(),
        "PYPROJECT_EOF",
        "",
        "# Ensure uv is available",
        "if ! command -v uv &>/dev/null; then",
        '    echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"',
        "    exit 1",
        "fi",
        "",
        f'echo "=== Setting up {engine} environment in $ENV_DIR ==="',
        "",
        "# Remove stale lockfile so uv regenerates it from current pyproject.toml",
        "rm -f uv.lock",
        "",
    ]

    # vLLM needs a wheelhouse with renamed wheel (glibc 2.34 vs manylinux_2_35)
    if engine == "vllm":
        lines += [
            "# SDCC glibc 2.34 < manylinux_2_35 required by the vLLM PyPI wheel.",
            "# Download the wheel and rename its platform tag so uv accepts it.",
            "mkdir -p wheelhouse",
            "if ! ls wheelhouse/vllm-*manylinux_2_17* &>/dev/null; then",
            '    echo "Downloading vLLM wheel to wheelhouse..."',
            "    # Resolve download URL via PyPI JSON API",
            '    VLLM_URL=$(python3 -c "',
            "import urllib.request, json",
            "resp = urllib.request.urlopen('https://pypi.org/pypi/vllm/json')",
            "data = json.loads(resp.read())",
            "for u in data['urls']:",
            "    if 'manylinux_2_35' in u['filename'] and 'x86_64' in u['filename']:",
            "        print(u['url']); break",
            '")',
            '    WHEEL_NAME=$(basename "$VLLM_URL")',
            '    curl -fSL "$VLLM_URL" -o "wheelhouse/$WHEEL_NAME"',
            "    # Rename manylinux_2_35 → manylinux_2_17 to bypass glibc check",
            '    WHEEL_FIXED="${WHEEL_NAME/manylinux_2_35/manylinux_2_17}"',
            '    mv "wheelhouse/$WHEEL_NAME" "wheelhouse/$WHEEL_FIXED"',
            '    echo "Renamed → $WHEEL_FIXED"',
            "fi",
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
            "uv pip install --no-deps --python .venv/bin/python wheelhouse/vllm-*manylinux_2_17*.whl",
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
            '"$PYTHON" -c "import importlib.metadata as m; print(f\'vLLM {m.version(\\"vllm\\")} OK\')"',
            '"$PYTHON" -c "import importlib.metadata as m; print(f\'PyTorch {m.version(\\"torch\\")}\')"',
            '"$PYTHON" -c "import importlib.metadata as m; print(f\'transformers {m.version(\\"transformers\\")}\')"',
        ]
    elif engine == "sglang":
        lines += [
            '"$PYTHON" -c "import importlib.metadata as m; print(f\'SGLang {m.version(\\"sglang\\")} OK\')"',
            '"$PYTHON" -c "import importlib.metadata as m; print(f\'PyTorch {m.version(\\"torch\\")}\')"',
        ]

    lines.append("")
    lines.append('echo "=== Setup complete ==="')
    lines.append("")

    script = "\n".join(lines)

    if dry_run:
        click.echo(script)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"Submitted setup job [bold]{job_id}[/] for [cyan]{engine}[/] engine."
    )
    console.print(f"  Environment: {env_dir}")
    console.print(f"  Monitor: squeue -j {job_id}")
    console.print(f"  Logs: ambix-setup-{engine}-{job_id}.log")
