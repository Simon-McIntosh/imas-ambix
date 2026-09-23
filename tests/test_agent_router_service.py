"""Standing router service submission tests."""

from __future__ import annotations

from click.testing import CliRunner

from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.cli import main


def test_router_script_is_cpu_only_and_uses_file_backed_admission(tmp_path):
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
    assert "#SBATCH --time=7-00:00:00" in script
    assert str(site.python_path("vllm")) in script
    assert "from imas_ambix.cli import main; main()" in script
    assert "agent router" in script
    assert "--api-key" not in script
    # Admission comes from router-gate.json rather than launch-only flags.
    assert "--max-in-flight" not in script
    assert "--max-queued" not in script


def test_lane_refresher_script_has_finite_time_limit(tmp_path):
    site = SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )

    script = slurm_mod.generate_lane_refresher_script(
        site,
        origin="http://127.0.0.1:18802",
    )

    assert "#SBATCH --time=7-00:00:00" in script


def test_engine_serve_script_remains_unlimited(tmp_path):
    site = SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )

    script = slurm_mod.generate_serve_script(
        load_profile("deepseek-v4-1-flash"),
        site,
    )

    assert "#SBATCH --time=0" in script


def test_router_dry_run_needs_no_admission_flags(monkeypatch):
    """The operator file keeps admission independent of launch composition."""

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
