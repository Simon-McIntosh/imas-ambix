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
