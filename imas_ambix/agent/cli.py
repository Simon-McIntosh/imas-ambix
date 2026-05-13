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


def _resolve_api_key(cli_value: str | None) -> str | None:
    """Resolve API key: CLI flag > envvar > .env file."""
    if cli_value:
        return cli_value
    env_key = os.environ.get("AMBIX_AGENT_API_KEY")
    if env_key:
        return env_key
    _load_dotenv()
    return os.environ.get("AMBIX_AGENT_API_KEY")


def _default_profile() -> str | None:
    """Resolve default profile: envvar > pyproject.toml [tool.ambix.agent]."""
    env_val = os.environ.get("AMBIX_AGENT_DEFAULT_PROFILE")
    if env_val:
        return env_val
    # Walk up from cwd to find pyproject.toml
    for parent in [Path.cwd(), *Path.cwd().parents]:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            import tomllib

            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                return (
                    data.get("tool", {})
                    .get("ambix", {})
                    .get("agent", {})
                    .get("default_profile")
                )
            except Exception:
                return None
    return None


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
def serve(slug: str | None, dry_run: bool, port: int | None) -> None:
    """Generate and submit a model serving job."""
    from imas_ambix.agent.slurm import generate_serve_script, submit_script

    profile = _load_profile(slug)
    site = SiteConfig.from_env()
    resolved_port = port if port is not None else site.default_port
    script = generate_serve_script(profile, site, port=resolved_port)

    if dry_run:
        click.echo(script)
        return

    try:
        job_id = submit_script(script)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"Submitted serve job {job_id} for {profile.slug} on port {resolved_port}."
    )


@agent.command()
def status() -> None:
    """Show active Ambix agent SLURM jobs."""
    user = os.environ.get("USER") or getpass.getuser()
    command = (
        "squeue -u "
        f"{shlex.quote(user)} "
        '-o "%.10i %.20j %.8T %.10M %.6D %R" | grep ambix- || true'
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
    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%i|%j"],
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
        if not job_name.startswith("ambix-"):
            continue
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
            prefix = f"ambix-serve-{resolved}"
            targets = [(jid, jn) for jid, jn in jobs if jn.startswith(prefix)]
        else:
            # No slug and no default — cancel all serve jobs
            targets = [(jid, jn) for jid, jn in jobs if jn.startswith("ambix-serve-")]

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
        raise click.ClickException(
            cancel_result.stderr.strip() or "scancel failed"
        )
    console.print(f"[green]Cancelled {len(job_ids)} job(s).[/]")


@agent.command()
@click.argument("slug", required=False)
@click.option("--url", default=None, help="Server base URL (e.g., http://host:8000).")
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
@click.option("--json", "json_output", is_flag=True, help="Output raw JSON.")
@click.option(
    "--output", "output_path", type=click.Path(), help="Write JSON results to file."
)
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
    warmup: bool,
) -> None:
    """Comprehensive LLM benchmark suite.

    Run against a profile: ambix agent bench deepseek-v4-flash

    Run against any endpoint: ambix agent bench --url http://host:8000 --model my-model
    """
    import json as json_mod
    import urllib.error
    import urllib.request

    from imas_ambix.agent.bench import BenchReport, _auth_headers, run_benchmark

    resolved_key = _resolve_api_key(api_key)

    # Resolve base_url and model
    if slug:
        profile = _load_profile(slug)
        base_url = url or "http://localhost:8000"
        model = model_name or profile.model.served_name
    elif url:
        base_url = url
        model = model_name or "default"
    else:
        # Try default profile
        default = _default_profile()
        if default:
            profile = _load_profile(default)
            base_url = url or "http://localhost:8000"
            model = model_name or profile.model.served_name
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
        raise click.ClickException(
            f"Cannot reach server at {base_url}: {exc}"
        ) from exc

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

    # JSON output
    if json_output:
        click.echo(report.to_json())
        if output_path:
            with open(output_path, "w") as f:
                f.write(report.to_json())
        return

    if output_path:
        with open(output_path, "w") as f:
            f.write(report.to_json())
        console.print(f"Results written to {output_path}")

    # Rich per-category tables
    from rich.table import Table as RichTable

    grouped: dict[str, list] = {}
    for r in report.results:
        grouped.setdefault(r.category, []).append(r)

    for cat, results in grouped.items():
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
            status_str = {
                "passed": "[green]✓[/]",
                "failed": "[red]✗[/]",
                "skipped": "[yellow]⊘[/]",
                "error": "[red]E[/]",
            }.get(r.status, r.status)
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

    # Summary table
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
            'mkdir -p wheelhouse',
            'if ! ls wheelhouse/vllm-*manylinux_2_17* &>/dev/null; then',
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
            '    # Rename manylinux_2_35 → manylinux_2_17 to bypass glibc check',
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
            '# --no-deps: all dependencies already installed by uv sync above',
            'uv pip install --no-deps --python .venv/bin/python wheelhouse/vllm-*manylinux_2_17*.whl',
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
            "\"$PYTHON\" -c \"import importlib.metadata as m; print(f'vLLM {m.version(\\\"vllm\\\")} OK')\"",
            "\"$PYTHON\" -c \"import importlib.metadata as m; print(f'PyTorch {m.version(\\\"torch\\\")}')\"",
            "\"$PYTHON\" -c \"import importlib.metadata as m; print(f'transformers {m.version(\\\"transformers\\\")}')\"",
        ]
    elif engine == "sglang":
        lines += [
            "\"$PYTHON\" -c \"import importlib.metadata as m; print(f'SGLang {m.version(\\\"sglang\\\")} OK')\"",
            "\"$PYTHON\" -c \"import importlib.metadata as m; print(f'PyTorch {m.version(\\\"torch\\\")}')\"",
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
