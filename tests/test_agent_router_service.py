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
        max_in_flight=7,
        max_queued=11,
        retry_after_seconds=13,
    )

    assert "#SBATCH --gres=gpu" not in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "#SBATCH --mem=12G" in script
    assert "#SBATCH --comment=ambix-router;port=18802" in script
    assert "#SBATCH --time=0" in script
    assert str(site.python_path("vllm")) in script
    assert "from imas_ambix.cli import main; main()" in script
    assert "agent router" in script
    assert "--max-in-flight 7" in script
    assert "--max-queued 11" in script
    assert "--retry-after-seconds 13" in script
    assert "--api-key" not in script


def test_router_dry_run_passes_submission_admission_limits(monkeypatch):
    def refuse_submission(_script: str) -> str:
        raise AssertionError("dry-run must not submit")

    monkeypatch.setattr(slurm_mod, "submit_script", refuse_submission)
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "router",
            "--port",
            "18802",
            "--dry-run",
            "--max-in-flight",
            "8",
            "--max-queued",
            "12",
            "--retry-after-seconds",
            "14",
        ],
    )

    assert result.exit_code == 0
    assert "--max-in-flight 8" in result.output
    assert "--max-queued 12" in result.output
    assert "--retry-after-seconds 14" in result.output


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
