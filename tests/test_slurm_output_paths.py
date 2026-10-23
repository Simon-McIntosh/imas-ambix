"""SLURM stdout stays in the user's state directory."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from imas_ambix.agent import cli as agent_cli
from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.profile import SiteConfig, load_profile
from imas_ambix.agent.slurm import (
    generate_download_script,
    generate_lane_refresher_script,
    generate_router_script,
    generate_serve_script,
)
from imas_ambix.cli import main
from imas_ambix.data.cli import _render_slurm_script


def _output_paths(script: str) -> list[Path]:
    outputs = [
        line.partition("=")[2]
        for line in script.splitlines()
        if line.startswith("#SBATCH --output=")
    ]
    assert outputs
    return [Path(output) for output in outputs]


@pytest.mark.parametrize(
    "script_kind",
    ("serve", "lane", "router", "download", "runtime-check", "setup", "data"),
)
def test_generated_scripts_use_absolute_state_log_path(
    script_kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    site = SiteConfig()
    profile = load_profile("glm-5-3")
    scripts = {
        "serve": lambda: generate_serve_script(profile, site),
        "lane": lambda: generate_lane_refresher_script(site, origin="http://localhost"),
        "router": lambda: generate_router_script(site, port=18802),
        "download": lambda: generate_download_script(profile, site),
        "runtime-check": lambda: agent_cli._engine_runtime_check_script(
            "vllm", site, "42", "identity"
        ),
        "setup": lambda: (
            CliRunner().invoke(main, ["agent", "setup", "vllm", "--dry-run"]).output
        ),
        "data": lambda: _render_slurm_script(
            partition="sun_debug",
            time_limit="00:30:00",
            dest=tmp_path / "mirror",
            manifest_path="manifest.json",
            numworkers=2,
        ),
    }
    outputs = _output_paths(scripts[script_kind]())
    assert len(outputs) == (2 if script_kind == "setup" else 1)
    for output in outputs:
        assert output.is_absolute()
        assert output.parent == tmp_path / ".local/state/ambix/slurm-logs"
    assert not (tmp_path / ".local/state/ambix/slurm-logs").exists()


@pytest.mark.parametrize(
    "script_name",
    (
        "spine_label_fleet.sh",
        "temporal_operator_cycle.sh",
        "run_magnetics_era_scorecard.sbatch",
    ),
)
def test_tracked_batch_scripts_use_state_log_path(script_name: str) -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / script_name).read_text()
    output = _output_paths(script)
    assert len(output) == 1
    assert output[0].is_absolute()
    assert output[0].parent == Path.home() / ".local/state/ambix/slurm-logs"


def test_setup_submission_names_both_state_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    submitted = []
    log_dir = tmp_path / ".local/state/ambix/slurm-logs"
    assert not log_dir.exists()

    def submit(script: str) -> str:
        assert log_dir.is_dir()
        submitted.append(script)
        if len(submitted) == 1:
            log_dir.rmdir()
        return str(40 + len(submitted))

    monkeypatch.setattr("imas_ambix.agent.slurm.submit_script", submit)
    result = CliRunner().invoke(main, ["agent", "setup", "vllm"])

    assert result.exit_code == 0, result.output
    message = result.output.replace("\n", "")
    assert str(log_dir / "ambix-setup-vllm-41.log") in message
    assert str(log_dir / "ambix-runtime-check-vllm-42.log") in message
    assert len(submitted) == 2
    assert log_dir.is_dir()


def test_submit_creates_log_directory_before_sbatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".local/state/ambix/slurm-logs"
    assert not log_dir.exists()

    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        assert argv == ["sbatch", "--parsable"]
        assert kwargs["input"] == "#!/bin/bash"
        assert log_dir.is_dir()
        return SimpleNamespace(returncode=0, stdout="31415\n", stderr="")

    monkeypatch.setattr(slurm_mod, "subprocess", SimpleNamespace(run=run))

    assert slurm_mod.submit_script("#!/bin/bash") == "31415"
    assert log_dir.is_dir()
