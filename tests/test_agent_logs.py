"""Tests for reading scheduler-owned Ambix job logs."""

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from imas_ambix.cli import main


def _job(
    job_id: str,
    name: str,
    *,
    state: str = "RUNNING",
    comment: str = "ambix-serve;port=18800",
) -> dict[str, str]:
    return {
        "jobid": job_id,
        "name": name,
        "state": state,
        "time": "1:00",
        "node": "gpu-node",
        "gres": "gpu:2",
        "comment": comment,
    }


def _patch_log_lookup(monkeypatch, jobs, path: Path | None):
    from imas_ambix.agent import cli as cli_mod

    calls = []

    def running_jobs(site, *, job_ids=()):
        calls.append(job_ids)
        return jobs

    monkeypatch.setattr(cli_mod, "_running_jobs", running_jobs)
    monkeypatch.setattr(
        cli_mod,
        "_job_stdout_path",
        lambda job_id: path,
        raising=False,
    )
    return calls


def test_job_stdout_path_comes_from_scheduler_metadata(monkeypatch):
    from imas_ambix.agent import cli as cli_mod

    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="JobId=42 JobName=glm-5-3 StdOut=/logs/glm-5-3-42.log",
            stderr="",
        )

    monkeypatch.setattr(cli_mod.subprocess, "run", run)

    assert cli_mod._job_stdout_path("42") == Path("/logs/glm-5-3-42.log")
    assert seen == [["scontrol", "show", "job", "-o", "42"]]


def test_logs_resolves_serve_from_profile_slug(monkeypatch, tmp_path):
    log_path = tmp_path / "serve.log"
    log_path.write_text("ready\nrequest complete\n", encoding="utf-8")
    calls = _patch_log_lookup(
        monkeypatch,
        [_job("42", "deepseek-v4-flash")],
        log_path,
    )

    result = CliRunner().invoke(
        main,
        ["agent", "logs", "deepseek-v4-flash"],
    )

    assert result.exit_code == 0
    assert calls == [()]
    assert str(log_path) in result.output
    assert "ready" in result.output
    assert "request complete" in result.output


def test_logs_without_selector_uses_default_profile(monkeypatch, tmp_path):
    from imas_ambix.agent import cli as cli_mod

    log_path = tmp_path / "default.log"
    log_path.write_text("default serve\n", encoding="utf-8")
    _patch_log_lookup(monkeypatch, [_job("43", "glm-5-3")], log_path)
    monkeypatch.setattr(cli_mod, "_default_profile", lambda: "glm-5-3")

    result = CliRunner().invoke(main, ["agent", "logs"])

    assert result.exit_code == 0
    assert "default serve" in result.output


def test_logs_resolves_router_from_job_id(monkeypatch, tmp_path):
    log_path = tmp_path / "router.log"
    log_path.write_text("route selected\n", encoding="utf-8")
    calls = _patch_log_lookup(
        monkeypatch,
        [_job("271", "ambix-router", comment="ambix-router;port=18808")],
        log_path,
    )

    result = CliRunner().invoke(main, ["agent", "logs", "271"])

    assert result.exit_code == 0
    assert calls == [("271",)]
    assert str(log_path) in result.output
    assert "route selected" in result.output


def test_logs_honours_trailing_line_limit(monkeypatch, tmp_path):
    log_path = tmp_path / "large.log"
    log_path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    _patch_log_lookup(monkeypatch, [_job("44", "glm-5-3")], log_path)

    result = CliRunner().invoke(
        main,
        ["agent", "logs", "glm-5-3", "--lines", "2"],
    )

    assert result.exit_code == 0
    assert "one" not in result.output
    assert "two" not in result.output
    assert "three" in result.output
    assert "four" in result.output


def test_logs_can_follow_only_new_output(monkeypatch, tmp_path):
    import time

    log_path = tmp_path / "growing.log"
    log_path.write_text("historical\n", encoding="utf-8")
    _patch_log_lookup(monkeypatch, [_job("46", "glm-5-3")], log_path)
    appended = False

    def grow_log(delay):
        nonlocal appended
        if appended:
            raise KeyboardInterrupt
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("new output\n")
        appended = True

    monkeypatch.setattr(time, "sleep", grow_log)

    result = CliRunner().invoke(
        main,
        ["agent", "logs", "glm-5-3", "--lines", "0", "--follow"],
    )

    assert result.exit_code == 0
    assert "historical" not in result.output
    assert "new output" in result.output


def test_logs_reports_no_such_job(monkeypatch):
    _patch_log_lookup(monkeypatch, [], None)

    result = CliRunner().invoke(main, ["agent", "logs", "999"])

    assert result.exit_code != 0
    assert "No such Ambix job" in result.output
    assert "999" in result.output


def test_logs_reports_job_has_not_produced_a_log(monkeypatch):
    _patch_log_lookup(monkeypatch, [_job("45", "glm-5-3")], None)

    result = CliRunner().invoke(main, ["agent", "logs", "45"])

    assert result.exit_code != 0
    assert "has not produced a log yet" in result.output
    assert "45" in result.output


def test_logs_reports_recorded_path_is_no_longer_readable(monkeypatch, tmp_path):
    log_path = tmp_path / "removed" / "router.log"
    _patch_log_lookup(
        monkeypatch,
        [_job("272", "ambix-router", comment="ambix-router;port=18808")],
        log_path,
    )

    result = CliRunner().invoke(main, ["agent", "logs", "272"])

    assert result.exit_code != 0
    assert "recorded log is no longer readable" in result.output
    assert str(log_path) in result.output
