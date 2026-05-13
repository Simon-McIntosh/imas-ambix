"""CLI commands for managing Ambix agent deployments."""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess

import click
from rich.console import Console
from rich.table import Table

from imas_ambix.agent.profile import SiteConfig, list_profiles, load_profile

console = Console()

# Engine types that have uv-managed environments
ENGINE_TYPES = ("vllm", "sglang")


def _load_profile(slug: str):
    try:
        return load_profile(slug)
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
@click.argument("slug")
def info(slug: str) -> None:
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
@click.argument("slug")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SLURM script instead of submitting it.",
)
def download(slug: str, dry_run: bool) -> None:
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
@click.argument("slug")
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
def serve(slug: str, dry_run: bool, port: int | None) -> None:
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
@click.argument("slug")
@click.option(
    "--url",
    default=None,
    help="Base URL of the running server (default: http://localhost:8000).",
)
@click.option(
    "--preset",
    type=click.Choice(["short", "medium", "long", "code", "thinking", "all"]),
    default="all",
    help="Benchmark preset to run.",
)
@click.option(
    "--repeat",
    type=int,
    default=1,
    help="Number of times to repeat each preset.",
)
@click.option(
    "--tool-test",
    is_flag=True,
    help="Include tool-call capability test.",
)
def bench(slug: str, url: str | None, preset: str, repeat: int, tool_test: bool) -> None:
    """Benchmark a running model server (TPS, TTFT, throughput)."""
    from rich.table import Table as RichTable

    from imas_ambix.agent.bench import (
        BENCH_PRESETS,
        BenchSuite,
        run_bench_preset,
        run_tool_call_bench,
    )

    profile = _load_profile(slug)
    base_url = url or "http://localhost:8000"
    model = profile.model.served_name

    # Quick health check
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
            models_data = json.loads(resp.read())
            available = [m["id"] for m in models_data.get("data", [])]
            if model not in available:
                console.print(
                    f"[yellow]Warning:[/] model '{model}' not in server models: {available}"
                )
                if available:
                    model = available[0]
                    console.print(f"Using '{model}' instead.")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise click.ClickException(
            f"Cannot reach server at {base_url}: {exc}"
        ) from exc

    console.print(f"\n[bold]Benchmarking[/] {model} at {base_url}\n")

    presets_to_run = (
        [p for p in BENCH_PRESETS if p != "tool_use"]
        if preset == "all"
        else [preset]
    )

    all_suites: list[tuple[str, BenchSuite]] = []
    for p in presets_to_run:
        desc = BENCH_PRESETS[p]["description"]
        console.print(f"  ▸ {p}: {desc} (×{repeat})")
        suite = run_bench_preset(base_url, model, p, repeat=repeat)
        all_suites.append((p, suite))
        for r in suite.results:
            if r.ok:
                console.print(
                    f"    ✓ {r.completion_tokens} tokens in {r.total_time_s:.1f}s "
                    f"({r.tokens_per_second:.1f} tok/s, "
                    f"TTFT {r.time_to_first_token_s * 1000:.0f}ms)"
                )
            else:
                console.print(f"    ✗ {r.error}")

    # Tool-call test
    if tool_test or preset == "all":
        console.print("  ▸ tool_use: Tool calling test")
        tool_result = run_tool_call_bench(base_url, model)
        if tool_result.ok:
            console.print(
                f"    ✓ Tool call succeeded in {tool_result.total_time_s:.1f}s"
            )
        else:
            console.print(f"    ✗ {tool_result.error}")

    # Summary table
    console.print()
    table = RichTable(title=f"Benchmark Summary — {model}")
    table.add_column("Preset", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("TPS", justify="right", style="bold green")
    table.add_column("TTFT (ms)", justify="right")

    for name, suite in all_suites:
        s = suite.summary()
        if s.get("passed", 0) > 0:
            table.add_row(
                name,
                str(s["total_tokens"]),
                str(s["total_time_s"]),
                str(s["avg_tps"]),
                str(s["avg_ttft_ms"]),
            )
        else:
            table.add_row(name, "—", "—", "FAILED", "—")

    console.print(table)


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
