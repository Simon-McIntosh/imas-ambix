"""Regression coverage for refreshing the deployment-derived clive settings."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "refresh_clive_context_window.py"


@pytest.fixture
def refresh_module() -> ModuleType:
    """Load the standalone script as a testable module."""
    spec = importlib.util.spec_from_file_location(
        "refresh_clive_context_window", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(tmp_path: Path, config: str, reservation: int = 32000) -> tuple[Path, Path]:
    flight = tmp_path / "flight.yaml"
    launcher = tmp_path / "clive"
    flight.write_text(config, encoding="utf-8")
    launcher.write_text(f"OUTPUT_RESERVATION={reservation}\n", encoding="utf-8")
    return flight, launcher


def _config() -> str:
    return """backends:
  unrelated:
    usable_input_window: 100
  clive:
    model: retired-checkpoint
    alias: Retired
    usable_input_window: 400000
    command: clive
"""


def _run(
    refresh_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    config: Path,
    launcher: Path,
    model: str = "deepseek-v4.1-flash",
    window: int = 204800,
    write: bool = False,
) -> int:
    monkeypatch.setattr(
        refresh_module, "advertised_deployment", lambda origin: (model, window)
    )
    arguments = ["--config", str(config), "--launcher", str(launcher)]
    if write:
        arguments.append("--write")
    return refresh_module.main(arguments)


def test_endpoint_identity_uses_the_served_model_id(
    monkeypatch: pytest.MonkeyPatch, refresh_module: ModuleType
) -> None:
    """The endpoint card supplies both the model identity and its context cap."""

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *unused: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data":[{"id":"served-checkpoint","max_model_len":204800}]}'

    monkeypatch.setattr(
        refresh_module.urllib.request, "urlopen", lambda url, timeout: Response()
    )

    assert refresh_module.advertised_deployment("http://fixture.invalid") == (
        "served-checkpoint",
        204800,
    )


@pytest.mark.parametrize(
    ("model", "alias"),
    [
        ("deepseek-v4-flash", "dsv4-flash"),
        ("deepseek-v4.1-flash", "dsv4.1-flash"),
    ],
)
def test_declared_alias_uses_the_recorded_short_spelling(
    refresh_module: ModuleType, model: str, alias: str
) -> None:
    """The two attested clive display labels remain explicit mappings."""
    assert refresh_module.declared_alias(model) == alias


def test_dry_run_reports_all_deployment_changes_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    refresh_module: ModuleType,
) -> None:
    """Dry runs make every stale clive deployment field visible."""
    config, launcher = _paths(tmp_path, _config())
    original = config.read_text(encoding="utf-8")

    assert _run(refresh_module, monkeypatch, config, launcher) == 1

    assert config.read_text(encoding="utf-8") == original
    output = capsys.readouterr().out
    assert 'model               : retired-checkpoint -> "deepseek-v4.1-flash"' in output
    assert 'alias               : Retired -> "dsv4.1-flash"' in output
    assert "usable_input_window : 400000 -> 172800" in output


def test_write_reconciles_model_alias_and_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_module: ModuleType,
) -> None:
    """Applying the refresh updates the full clive deployment block."""
    config, launcher = _paths(tmp_path, _config())

    assert _run(refresh_module, monkeypatch, config, launcher, write=True) == 0

    updated = config.read_text(encoding="utf-8")
    assert 'model: "deepseek-v4.1-flash"' in updated
    assert 'alias: "dsv4.1-flash"' in updated
    assert "usable_input_window: 172800" in updated


def test_quarter_reservation_is_used_below_the_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_module: ModuleType,
) -> None:
    """The launcher-provided quarter reservation is not replaced by the cap."""
    config, launcher = _paths(tmp_path, _config(), reservation=25000)

    assert (
        _run(
            refresh_module,
            monkeypatch,
            config,
            launcher,
            window=100000,
            write=True,
        )
        == 0
    )

    assert "usable_input_window: 75000" in config.read_text(encoding="utf-8")


def test_window_rewrite_is_anchored_to_the_clive_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_module: ModuleType,
) -> None:
    """A same-named field in another backend cannot redirect the rewrite."""
    config, launcher = _paths(tmp_path, _config())

    assert _run(refresh_module, monkeypatch, config, launcher, write=True) == 0

    updated = config.read_text(encoding="utf-8")
    assert "unrelated:\n    usable_input_window: 100" in updated
    assert "clive:\n    model:" in updated
    assert "usable_input_window: 172800" in updated


def test_unmapped_model_refuses_without_touching_the_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_module: ModuleType,
) -> None:
    """A new served model needs an explicit display-name decision."""
    config, launcher = _paths(tmp_path, _config())
    original = config.read_text(encoding="utf-8")

    with pytest.raises(SystemExit, match="unmapped-checkpoint.*add a mapping"):
        _run(
            refresh_module,
            monkeypatch,
            config,
            launcher,
            model="unmapped-checkpoint",
            write=True,
        )

    assert config.read_text(encoding="utf-8") == original
