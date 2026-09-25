"""The gate control file is named explicitly, not inferred from the lane.

The router resolves its control file from the first source that names one: the
explicit option or ``AMBIX_ROUTER_GATE_PATH``, then the site control path
(``SiteConfig.gate_control_path``, defaulting to the group-owned control
directory under the project base), and only then the lane document's sibling.
The sibling step is the last resort, reached only when the site value is empty,
so a deployment that names nothing and sets the site value empty keeps taking
control from exactly where it did before the site default existed. The CLI
resolves the same path and stamps ``set_by``/``set_at`` so a group-writable file
stays auditable.
"""

from __future__ import annotations

import getpass
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from click.testing import CliRunner

from imas_ambix.agent import slurm as slurm_mod
from imas_ambix.agent.profile import GATE_CONTROL_PATH_ENV, SiteConfig
from imas_ambix.agent.router import (
    GATE_FILENAME,
    GATE_PATH_ENV,
    RouterApp,
    Upstream,
    resolve_gate_path,
)
from imas_ambix.cli import main

# The revision before the router script learned to carry the gate path. The
# no-option script is compared against it so the option is additive.
_PRE_GATE_PATH_OPTION_REVISION = "268df346"


class _Resolver:
    def __init__(self) -> None:
        self.upstreams = (Upstream(base_url="http://127.0.0.1:1", model_id="m"),)

    async def resolve(self) -> tuple[Upstream, ...]:
        return self.upstreams


def _gate_payload(width: int) -> str:
    return json.dumps({"width": width, "wait_seconds": 1.0})


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _control_and_lane(tmp_path: Path) -> tuple[Path, Path]:
    """A scratch control directory standing in for the group-owned control dir.

    The real control file lives under ``/work/projects/imas_gpu/agents/control``;
    a scratch directory plays that role here so the test never touches GPFS.
    """
    control = tmp_path / "control" / GATE_FILENAME
    lane = tmp_path / "public" / "lane.json"
    control.parent.mkdir(parents=True, exist_ok=True)
    lane.parent.mkdir(parents=True, exist_ok=True)
    return control, lane


def _emitted_gate_export(output: str) -> Path:
    """The gate path the CLI actually wrote, read back from the emitted script.

    Reading the value out of the output is the point: an assertion computed from
    the same expression the CLI uses cannot show that the CLI expanded anything,
    because both sides would move together.
    """
    marker = f"export {GATE_PATH_ENV}="
    line = next(line for line in output.splitlines() if line.startswith(marker))
    value = line[len(marker) :].split("#", 1)[0].strip()
    return Path(value.strip("'\""))


def test_resolve_gate_path_prefers_explicit_then_env_then_sibling(
    tmp_path, monkeypatch
) -> None:
    """The order is explicit, env, site control, then the lane sibling.

    The sibling step is the last resort and is reached here only because the
    site control value is set empty; with its default it would win the step
    before.
    """
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, "")
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    lane = tmp_path / "lane.json"
    explicit = tmp_path / "control" / GATE_FILENAME

    assert resolve_gate_path(explicit, lane) == explicit
    monkeypatch.setenv(GATE_PATH_ENV, str(explicit))
    assert resolve_gate_path(None, lane) == explicit
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    assert resolve_gate_path(None, lane) == lane.with_name(GATE_FILENAME)
    assert resolve_gate_path(None, None) is None


def test_router_takes_an_explicit_path_independent_of_the_lane_directory(
    tmp_path, monkeypatch
) -> None:
    """The explicit path wins even when a populated sibling also exists."""
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    control, lane = _control_and_lane(tmp_path)
    # A decoy sibling the router must NOT read: if the explicit path were
    # ignored this is the file it would take its width from.
    sibling = lane.with_name(GATE_FILENAME)
    sibling.write_text(_gate_payload(99), encoding="utf-8")
    control.write_text(_gate_payload(7), encoding="utf-8")
    lane.write_text("{}", encoding="utf-8")

    app = RouterApp(_Resolver(), lane_document=lane, gate_file=control)

    assert app._generation_gate.config_path == control
    assert app._generation_gate.settings().width == 7
    # The sibling's wider value is neither read nor written.
    assert _read(sibling)["width"] == 99


def test_router_environment_variable_names_the_control_path(
    tmp_path, monkeypatch
) -> None:
    control, lane = _control_and_lane(tmp_path)
    sibling = lane.with_name(GATE_FILENAME)
    sibling.write_text(_gate_payload(99), encoding="utf-8")
    control.write_text(_gate_payload(5), encoding="utf-8")
    lane.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(GATE_PATH_ENV, str(control))

    app = RouterApp(_Resolver(), lane_document=lane)

    assert app._generation_gate.config_path == control
    assert app._generation_gate.settings().width == 5


def test_router_falls_back_to_the_lane_sibling_when_neither_is_named(
    tmp_path, monkeypatch
) -> None:
    """With the site value empty, naming nothing keeps the lane sibling.

    The site control path precedes this branch, so this is the pre-default
    behaviour preserved: an existing deployment that names nothing and carries
    an empty site value keeps taking control from the lane document's sibling.
    """
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, "")
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    _, lane = _control_and_lane(tmp_path)
    monkeypatch.setenv(
        "AMBIX_AGENT_ENDPOINT_DOCUMENT", str(lane.parent / "endpoints.json")
    )
    sibling = lane.with_name(GATE_FILENAME)
    sibling.write_text(_gate_payload(13), encoding="utf-8")
    lane.write_text("{}", encoding="utf-8")

    app = RouterApp(_Resolver(), lane_document=lane)

    assert app._generation_gate.config_path == sibling
    assert app._generation_gate.settings().width == 13


def test_a_second_process_write_is_honoured_on_the_next_refresh(
    tmp_path, monkeypatch
) -> None:
    """A write to the explicit path reaches the gate within one refresh.

    The writer is a separate process, which is the case that matters: the
    operator who owns the group-writable control file is not the router, so the
    router has to pick the change up by re-reading the file rather than by any
    in-process signal.
    """
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    control, lane = _control_and_lane(tmp_path)
    control.write_text(_gate_payload(5), encoding="utf-8")
    lane.write_text("{}", encoding="utf-8")
    app = RouterApp(_Resolver(), lane_document=lane, gate_file=control)
    assert app._generation_gate.settings().width == 5

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; open(sys.argv[1], 'w').write("
            "json.dumps({'width': 11, 'wait_seconds': 1.0}))",
            str(control),
        ],
        check=True,
    )

    observed: list[int] = []
    for _ in range(400):  # >3 s, comfortably past the one-second refresh
        observed.append(app._generation_gate.settings().width)
        if observed[-1] == 11:
            break
        time.sleep(0.01)
    assert observed[-1] == 11, observed


def test_cli_writes_resolve_the_explicit_path_and_stamp_the_writer(
    tmp_path, monkeypatch
) -> None:
    """pause, resume and a width write share one path and one audit stamp."""
    control, lane = _control_and_lane(tmp_path)
    monkeypatch.setenv(GATE_PATH_ENV, str(control))
    monkeypatch.setenv(
        "AMBIX_AGENT_ENDPOINT_DOCUMENT", str(lane.parent / "endpoints.json")
    )
    lane.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")
    sibling = lane.with_name(GATE_FILENAME)

    paused = CliRunner().invoke(main, ["agent", "pause", "--reason", "relaunch"])
    assert paused.exit_code == 0, paused.output
    assert control.exists()
    assert not sibling.exists(), (
        "the write went to the lane sibling, not the named path"
    )
    written = _read(control)
    assert written["paused"] is True
    assert written["set_by"] == getpass.getuser()
    _assert_utc_iso(str(written["set_at"]))

    resized = CliRunner().invoke(main, ["agent", "width", "14"])
    assert resized.exit_code == 0, resized.output
    assert not sibling.exists()
    written = _read(control)
    assert written["width"] == 14
    assert written["set_by"] == getpass.getuser()
    _assert_utc_iso(str(written["set_at"]))

    resumed = CliRunner().invoke(main, ["agent", "resume"])
    assert resumed.exit_code == 0, resumed.output
    written = _read(control)
    assert written["paused"] is False
    assert written["set_by"] == getpass.getuser()
    _assert_utc_iso(str(written["set_at"]))


def test_cli_gate_file_option_overrides_the_environment(tmp_path, monkeypatch) -> None:
    control, lane = _control_and_lane(tmp_path)
    other = tmp_path / "elsewhere" / GATE_FILENAME
    other.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(GATE_PATH_ENV, str(control))
    monkeypatch.setenv(
        "AMBIX_AGENT_ENDPOINT_DOCUMENT", str(lane.parent / "endpoints.json")
    )
    lane.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")

    result = CliRunner().invoke(
        main, ["agent", "width", "9", "--gate-file", str(other)]
    )

    assert result.exit_code == 0, result.output
    assert other.exists()
    assert not control.exists()
    assert _read(other)["width"] == 9


def test_width_accepts_auto_and_refuses_other_text(tmp_path, monkeypatch) -> None:
    control, lane = _control_and_lane(tmp_path)
    monkeypatch.setenv(GATE_PATH_ENV, str(control))
    monkeypatch.setenv(
        "AMBIX_AGENT_ENDPOINT_DOCUMENT", str(lane.parent / "endpoints.json")
    )
    lane.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")

    auto = CliRunner().invoke(main, ["agent", "width", "auto"])
    assert auto.exit_code == 0, auto.output
    assert _read(control)["width"] == "auto"

    bad = CliRunner().invoke(main, ["agent", "width", "wide"])
    assert bad.exit_code != 0
    assert "neither an integer" in bad.output


def test_lane_document_publishes_the_explicit_gate_path(tmp_path, monkeypatch) -> None:
    """A reader can see which file the running router took control from."""
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    control, lane = _control_and_lane(tmp_path)
    control.write_text(_gate_payload(7), encoding="utf-8")
    lane.write_text(json.dumps({"running": 3}), encoding="utf-8")
    app = RouterApp(_Resolver(), lane_document=lane, gate_file=control)

    app._publish_gate_snapshot()

    published = _read(lane)["router_generation_gate"]
    assert isinstance(published, dict)
    assert published["config_path"] == str(control)


def _assert_utc_iso(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, value
    assert parsed.utcoffset().total_seconds() == 0, value


def _scratch_site(tmp_path: Path) -> SiteConfig:
    return SiteConfig(
        base_dir=str(tmp_path),
        engine_env_root=str(tmp_path / "engine-envs"),
    )


def _generate_at_revision(revision: str, site: SiteConfig, **kwargs) -> str:
    """Run the router-script generator as it stood at ``revision``.

    The base module is compiled with this module's own ``__file__`` so it
    resolves the repository root identically; otherwise the embedded
    ``PYTHONPATH`` would differ for a reason unrelated to the change under
    test.
    """
    source = subprocess.run(
        ["git", "show", f"{revision}:imas_ambix/agent/slurm.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    namespace: dict[str, object] = {
        "__file__": slurm_mod.__file__,
        "__name__": "base_slurm_under_test",
    }
    exec(compile(source, slurm_mod.__file__, "exec"), namespace)
    generate = namespace["generate_router_script"]
    return generate(site, **kwargs)  # type: ignore[operator]


def _without_unset_line(script: str) -> str:
    """The no-option script with its single unset line removed.

    The pre-option revision had no gate-path line at all, so stripping this one
    line recovers the script the generator produced before the option existed --
    the strongest available statement that the option changed nothing else.
    """
    lines = script.splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(f"unset {GATE_PATH_ENV}")]
    assert len(kept) == len(lines) - 1, script
    return "".join(kept)


def test_generated_script_without_the_option_unsets_the_gate_path(tmp_path) -> None:
    """A no-option submission reads the lane sibling, whatever the shell exported.

    The submitted job inherits the submitting shell's environment (no --export
    restriction is set), so a value exported there would otherwise decide which
    control file a no-option router reads. The generated script unsets it, which
    makes the no-option behaviour a property of the script alone.
    """
    site = _scratch_site(tmp_path)

    current = slurm_mod.generate_router_script(site, port=18802, cpus=3, memory="12G")
    base = _generate_at_revision(
        _PRE_GATE_PATH_OPTION_REVISION, site, port=18802, cpus=3, memory="12G"
    )
    assert f"unset {GATE_PATH_ENV}" in current
    assert f"export {GATE_PATH_ENV}=" not in current
    # The option is additive: the no-option script differs from the pre-option
    # revision by exactly the unset line, and by nothing else.
    assert _without_unset_line(current) == base

    probed = slurm_mod.generate_router_script(
        site, port=18802, cpus=3, memory="12G", prefix_probe=True
    )
    probed_base = _generate_at_revision(
        _PRE_GATE_PATH_OPTION_REVISION,
        site,
        port=18802,
        cpus=3,
        memory="12G",
        prefix_probe=True,
    )
    assert _without_unset_line(probed) == probed_base


def test_named_gate_path_is_embedded_in_the_generated_script(tmp_path) -> None:
    site = _scratch_site(tmp_path)
    control = tmp_path / "control" / GATE_FILENAME

    script = slurm_mod.generate_router_script(site, port=18802, gate_file=control)

    assert f"export {GATE_PATH_ENV}={control}" in script
    # The path is a value the running job carries, not one inferred later.
    assert str(control) in script.split("exec ", 1)[0]


def test_submitted_job_environment_inheritance_is_read_from_the_header(
    tmp_path,
) -> None:
    """State the inheritance fact from the header, never assume it."""
    site = _scratch_site(tmp_path)
    control = tmp_path / "control" / GATE_FILENAME

    script = slurm_mod.generate_router_script(site, port=18802, gate_file=control)

    export_lines = [
        line
        for line in script.splitlines()
        if line.startswith("#SBATCH") and "--export" in line
    ]
    # No --export restriction is set, so sbatch's default applies and the job
    # inherits the submitting environment. The gate path is exported in the body
    # anyway, so the job reads the chosen file regardless of what it inherits.
    assert export_lines == []
    assert any(
        line.startswith(f"export {GATE_PATH_ENV}=") for line in script.splitlines()
    )


def test_router_dry_run_shows_the_named_gate_path(tmp_path, monkeypatch) -> None:
    def refuse_submission(_script: str) -> str:
        raise AssertionError("dry-run must not submit")

    monkeypatch.setattr(slurm_mod, "submit_script", refuse_submission)
    control = tmp_path / "control" / GATE_FILENAME

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "router",
            "--port",
            "18802",
            "--dry-run",
            "--gate-file",
            str(control),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"export {GATE_PATH_ENV}={control}" in result.output


def test_router_submit_carries_the_gate_path_into_the_submitted_script(
    tmp_path, monkeypatch
) -> None:
    """The submit path hands the same script on, with no job submitted."""
    captured: dict[str, str] = {}

    def submit(script: str) -> str:
        captured["script"] = script
        return "4242"

    monkeypatch.setattr(slurm_mod, "submit_script", submit)
    control = tmp_path / "control" / GATE_FILENAME

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "router",
            "--port",
            "18802",
            "--submit",
            "--gate-file",
            str(control),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"export {GATE_PATH_ENV}={control}" in captured["script"]
    assert "Submitted keyless router job 4242" in result.output


def test_router_gate_file_tilde_is_absolute_under_the_home_directory(
    tmp_path, monkeypatch
) -> None:
    """A '~' argument reaches the generated script fully expanded.

    sbatch does not run the submitting shell, so a literal '~' in the exported
    line names a directory nobody meant and the submitted router would read the
    wrong file -- or none.
    """
    monkeypatch.setattr(slurm_mod, "submit_script", lambda _script: "4242")

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "router",
            "--port",
            "18802",
            "--dry-run",
            "--gate-file",
            "~/ambix-gate.json",
        ],
    )

    assert result.exit_code == 0, result.output
    # Assert on the path the CLI emitted, not on a value this test computed the
    # same way the CLI does: an unexpanded ``~`` reaching the script would leave
    # a re-derived expectation unchanged and pass.
    emitted = _emitted_gate_export(result.output)
    assert emitted.is_absolute(), emitted
    assert emitted.parent == Path.home().resolve(), emitted
    assert emitted == Path.home().resolve() / "ambix-gate.json"


def test_router_gate_file_relative_argument_resolves_against_the_cwd(
    tmp_path, monkeypatch
) -> None:
    """A relative argument becomes an absolute path, not a cwd-relative guess."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slurm_mod, "submit_script", lambda _script: "4242")

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "router",
            "--port",
            "18802",
            "--dry-run",
            "--gate-file",
            "control/router-gate.json",
        ],
    )

    assert result.exit_code == 0, result.output
    # The emitted path is read back from the script, so a relative value the CLI
    # failed to resolve fails this rather than matching a re-derived guess.
    emitted = _emitted_gate_export(result.output)
    assert emitted.is_absolute(), emitted
    assert emitted == (tmp_path / "control" / GATE_FILENAME).resolve()


def test_foreground_router_receives_the_expanded_absolute_gate_file(
    tmp_path, monkeypatch
) -> None:
    """The foreground router gets the same resolved path the script carries.

    A ``~`` is expanded once, in the CLI, before it reaches either the generated
    script or the running server, so the foreground router reads the file the
    operator named rather than one its own shell would resolve differently.
    """
    import imas_ambix.agent.router as router_mod

    captured: dict[str, object] = {}

    def fake_serve_router(_resolver, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(router_mod, "serve_router", fake_serve_router)

    result = CliRunner().invoke(
        main,
        ["agent", "router", "--port", "18802", "--gate-file", "~/ambix-gate.json"],
    )

    assert result.exit_code == 0, result.output
    gate_file = captured["gate_file"]
    assert gate_file is not None
    assert isinstance(gate_file, Path)
    assert gate_file.is_absolute(), gate_file
    assert gate_file == Path.home().resolve() / "ambix-gate.json"


def test_router_no_option_dry_run_carries_the_unset_line(tmp_path, monkeypatch) -> None:
    """A submission with no option does not inherit the shell's gate path."""
    monkeypatch.setattr(slurm_mod, "submit_script", lambda _script: "4242")

    result = CliRunner().invoke(
        main, ["agent", "router", "--port", "18802", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert f"unset {GATE_PATH_ENV}" in result.output
    assert f"export {GATE_PATH_ENV}=" not in result.output
