"""Tests for the count guard in the JT-60SA geometry figure script.

``scripts/figures_jt60sa_geometry.py`` reads each phase's geometry from the
description store, extracts it for imas-ink, and calls ``_report`` to print and
reconcile the two counts.  The guard exists so a silent extraction that drops
or duplicates geometry is caught before a figure is written; these tests drive
``_report`` directly, without running the script's ``main``, so the store is
never opened.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "figures_jt60sa_geometry.py"
)
_spec = importlib.util.spec_from_file_location(
    "figures_jt60sa_geometry_under_test", _MODULE_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

_report = _module._report


def _store(probes: int = 12) -> dict[str, int]:
    return {
        "pf_coils": 5,
        "filaments": 7,
        "first_wall_points": 100,
        "vessel_points": 200,
        "probes": probes,
        "flux_loops": 3,
    }


def _drawn(probes: int = 12) -> dict[str, int]:
    return {
        "pf_coils": 5,
        "first_wall_points": 100,
        "vessel_points": 200,
        "probes": probes,
        "flux_loops": 3,
    }


def test_report_accepts_matching_counts(capsys: pytest.CaptureFixture[str]) -> None:
    assert _report("OP1", _store(), _drawn()) is None


def test_report_refuses_probe_count_mismatch() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _report("OP1", _store(probes=12), _drawn(probes=11))
    assert "probes" in str(excinfo.value)
