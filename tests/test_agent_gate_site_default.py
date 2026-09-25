"""The gate control file defaults to the group control directory.

The router and the CLI resolve one control file through the same order: the
explicit option, then ``AMBIX_ROUTER_GATE_PATH``, then the site control path
(``SiteConfig.gate_control_path``, overridable through
``AMBIX_AGENT_GATE_CONTROL_PATH`` and defaulting to
``<base_dir>/agents/control/router-gate.json`` when the value is non-empty),
then the lane document's sibling. The site control path is where control lives
by default, in the group-owned directory under the project base, so the file is
not derived from wherever the lane publishes; the lane sibling is the last
resort and applies only when the site value is empty.

These tests pin that order, and that no test can reach the production path while
the session fixture holds the site value empty.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from imas_ambix.agent.profile import GATE_CONTROL_PATH_ENV, SiteConfig
from imas_ambix.agent.router import (
    GATE_FILENAME,
    GATE_PATH_ENV,
    RouterApp,
    Upstream,
    resolve_gate_path,
)
from imas_ambix.cli import main

_PRODUCTION_CONTROL_DIR = "/work/projects/imas_gpu/agents/control"


class _Resolver:
    def __init__(self) -> None:
        self.upstreams = (Upstream(base_url="http://127.0.0.1:1", model_id="m"),)

    async def resolve(self) -> tuple[Upstream, ...]:
        return self.upstreams


def _lane(tmp_path: Path) -> Path:
    lane = tmp_path / "public" / "lane.json"
    lane.parent.mkdir(parents=True, exist_ok=True)
    return lane


def test_site_config_defaults_the_control_path_under_the_base_dir(tmp_path) -> None:
    site = SiteConfig(base_dir=str(tmp_path))

    assert site.gate_control_path == str(
        tmp_path / "agents" / "control" / GATE_FILENAME
    )


def test_site_control_path_is_overridable_from_the_environment(
    tmp_path, monkeypatch
) -> None:
    elsewhere = tmp_path / "elsewhere" / GATE_FILENAME
    monkeypatch.setenv("AMBIX_AGENT_BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, str(elsewhere))

    assert SiteConfig.from_env().gate_control_path == str(elsewhere)


def test_site_default_beats_the_lane_document_sibling(tmp_path, monkeypatch) -> None:
    """The whole point of the default: control is not the lane document's sibling."""
    control = tmp_path / "control" / GATE_FILENAME
    lane = _lane(tmp_path)
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, str(control))
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)

    assert resolve_gate_path(None, lane) == control
    assert resolve_gate_path(None, lane) != lane.with_name(GATE_FILENAME)

    app = RouterApp(_Resolver(), lane_document=lane)
    assert app._generation_gate.config_path == control


def test_router_gate_path_environment_beats_the_site_default(
    tmp_path, monkeypatch
) -> None:
    site = tmp_path / "control" / GATE_FILENAME
    named = tmp_path / "explicit" / GATE_FILENAME
    lane = _lane(tmp_path)
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, str(site))
    monkeypatch.setenv(GATE_PATH_ENV, str(named))

    assert resolve_gate_path(None, lane) == named


def test_explicit_option_beats_every_other_source(tmp_path, monkeypatch) -> None:
    site = tmp_path / "control" / GATE_FILENAME
    named = tmp_path / "explicit" / GATE_FILENAME
    chosen = tmp_path / "chosen" / GATE_FILENAME
    lane = _lane(tmp_path)
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, str(site))
    monkeypatch.setenv(GATE_PATH_ENV, str(named))

    assert resolve_gate_path(chosen, lane) == chosen


def test_empty_site_value_falls_back_to_the_lane_sibling(
    tmp_path, monkeypatch
) -> None:
    """A deployment that names nothing is where it was before the default."""
    lane = _lane(tmp_path)
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, "")
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)

    assert resolve_gate_path(None, lane) == lane.with_name(GATE_FILENAME)
    assert resolve_gate_path(None, None) is None


def test_the_session_fixture_hides_the_production_control_path(
    tmp_path, monkeypatch
) -> None:
    """The guarded thing is made to happen: releasing it exposes the default.

    The first half is the assertion the fixture exists for. The second half is
    its positive control -- without the fixture's empty site value the
    production default appears, so the guard is not passing vacuously.
    """
    lane = _lane(tmp_path)

    assert "/work" not in str(resolve_gate_path(None, lane))

    monkeypatch.delenv(GATE_CONTROL_PATH_ENV, raising=False)
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    monkeypatch.delenv("AMBIX_AGENT_BASE_DIR", raising=False)
    released = resolve_gate_path(None, lane)

    assert released == Path(SiteConfig.from_env().gate_control_path)
    assert str(released).startswith(_PRODUCTION_CONTROL_DIR)


def test_cli_writes_the_site_control_file_when_no_option_is_given(
    tmp_path, monkeypatch
) -> None:
    """The CLI and the router resolve the same file, so the CLI honours it too."""
    control = tmp_path / "control" / GATE_FILENAME
    control.parent.mkdir(parents=True, exist_ok=True)
    lane = _lane(tmp_path)
    lane.write_text(json.dumps({"running": 0, "waiting": 0}), encoding="utf-8")
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, str(control))
    monkeypatch.delenv(GATE_PATH_ENV, raising=False)
    monkeypatch.setenv(
        "AMBIX_AGENT_ENDPOINT_DOCUMENT", str(lane.parent / "endpoints.json")
    )

    result = CliRunner().invoke(main, ["agent", "width", "9"])

    assert result.exit_code == 0, result.output
    assert json.loads(control.read_text(encoding="utf-8"))["width"] == 9
    assert not lane.with_name(GATE_FILENAME).exists()
