"""Command-line interface for imas-ambix."""

from __future__ import annotations

import importlib

import click


class _LazyGroup(click.Group):
    def __init__(self, import_path: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._import_path = import_path
        self._loaded_group: click.Group | None = None

    def _load(self) -> click.Group:
        if self._loaded_group is None:
            module_name, attr_name = self._import_path.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            self._loaded_group = getattr(module, attr_name)
        return self._loaded_group

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        return self._load().get_command(ctx, cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return self._load().list_commands(ctx)

    def invoke(self, ctx: click.Context) -> object:
        return self._load().invoke(ctx)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        self._load().format_help(ctx, formatter)


@click.group()
@click.version_option(package_name="imas-ambix")
def main() -> None:
    """Ambix — Fusion World Model training framework."""


@main.command()
def status() -> None:
    """Show training pipeline status."""
    click.echo("⚗️  ambix: no active training runs")


agent = _LazyGroup(
    import_path="imas_ambix.agent.cli:agent",
    name="agent",
    help="Manage LLM agent deployments on SLURM GPU clusters.",
)
main.add_command(agent)


data = _LazyGroup(
    import_path="imas_ambix.data.cli:data",
    name="data",
    help="FAIR-MAST data acquisition and access (see plans/data-acquisition.md).",
)
main.add_command(data)


tokenize = _LazyGroup(
    import_path="imas_ambix.tokenizer.cli:tokenize",
    name="tokenize",
    help="Multi-modal tokenizers for FAIR-MAST shots (see plans/tokenizers.md).",
)
main.add_command(tokenize)
