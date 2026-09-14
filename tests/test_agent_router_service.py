"""Standing router service submission tests."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.profile import SiteConfig
from imas_ambix.cli import main


def test_router_script_is_cpu_only_discoverable_and_unlimited(tmp_path):
    site = SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )

    script = slurm_mod.generate_router_script(
        site,
        port=18802,
        cpus=3,
        memory="12G",
    )

    assert "#SBATCH --gres=gpu" not in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "#SBATCH --mem=12G" in script
    assert "#SBATCH --comment=ambix-router;port=18802" in script
    assert "#SBATCH --time=0" in script
    assert str(site.python_path("vllm")) in script
    assert "from imas_ambix.cli import main; main()" in script
    assert "agent router" in script
    assert "--api-key" not in script
    # The engine owns scheduling; the router must not carry a second bound.
    assert "--max-in-flight" not in script
    assert "--max-queued" not in script


def test_router_dry_run_emits_no_request_bound(monkeypatch):
    """The relay must never reimpose a ceiling the engine already schedules."""

    def refuse_submission(_script: str) -> str:
        raise AssertionError("dry-run must not submit")

    monkeypatch.setattr(slurm_mod, "submit_script", refuse_submission)
    result = CliRunner().invoke(
        main,
        ["agent", "router", "--port", "18802", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "agent router" in result.output
    for retired in ("--max-in-flight", "--max-queued", "--retry-after-seconds"):
        assert retired not in result.output


def test_router_submit_uses_shared_submission_adapter(monkeypatch):
    captured: dict[str, str] = {}

    def submit(script: str) -> str:
        captured["script"] = script
        return "77"

    monkeypatch.setattr(slurm_mod, "submit_script", submit)
    result = CliRunner().invoke(
        main,
        ["agent", "router", "--port", "18802", "--submit"],
    )

    assert result.exit_code == 0
    assert "Submitted keyless router job 77" in result.output
    assert "#SBATCH --comment=ambix-router;port=18802" in captured["script"]
