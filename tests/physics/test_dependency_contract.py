"""Dependency and ownership checks for the Nova compatibility facade."""

import json
import subprocess
import sys
import tomllib
from importlib.metadata import distribution
from pathlib import Path

from nova.biot.coupling import CircuitCoupling as NovaCircuitCoupling
from nova.circuit import PassiveCircuitSystem as NovaPassiveCircuitSystem
from nova.equilibrium import ReconstructProfile as NovaReconstructProfile
from nova.equilibrium.harmonic import HarmonicConfig as NovaHarmonicConfig
from nova.equilibrium.moment import MomentConfig as NovaMomentConfig
from nova.transport import (
    CurrentDiffusion as NovaCurrentDiffusion,
)
from nova.transport import (
    traced_assemble_flux_surface_geometry as nova_traced_assemble_geometry,
)
from nova.transport import (
    traced_flux_surface_geometry as nova_traced_geometry,
)

from imas_ambix import physics
from imas_ambix.fluxstate.adapters import REVIEWED_CURRENT_DIFFUSION_REVISION
from imas_ambix.statespace.nova_ensemble_estimator import NOVA_REVISION

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from distribution_sources import editable_checkout  # noqa: E402

ROOT = Path(__file__).parents[2]


def test_project_and_lock_declare_editable_nova_checkout():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["requires-python"] == ">=3.14"
    assert "nova-stella" in project["project"]["dependencies"]
    assert project["tool"]["uv"]["sources"]["nova-stella"] == {
        "path": "../nova",
        "editable": True,
    }

    lock = (ROOT / "uv.lock").read_text()
    assert 'requires-python = ">=3.14"' in lock
    assert 'name = "nova-stella"' in lock
    assert 'source = { editable = "../nova" }' in lock


def test_facade_types_are_owned_by_nova():
    assert physics.ReconstructProfile is NovaReconstructProfile
    assert physics.MomentConfig is NovaMomentConfig
    assert physics.HarmonicConfig is NovaHarmonicConfig
    assert physics.PassiveCircuitSystem is NovaPassiveCircuitSystem
    assert physics.CircuitCoupling is NovaCircuitCoupling
    assert physics.CurrentDiffusion is NovaCurrentDiffusion
    assert (
        physics.traced_assemble_flux_surface_geometry is nova_traced_assemble_geometry
    )
    assert physics.traced_flux_surface_geometry is nova_traced_geometry


def test_installed_nova_distribution_reports_checkout_revision():
    installed = distribution("nova-stella")
    direct_url_json = installed.read_text("direct_url.json")
    direct_url = json.loads(direct_url_json)
    assert direct_url["dir_info"]["editable"] is True
    source = editable_checkout(direct_url_json)
    assert source is not None
    resolved = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolved == NOVA_REVISION == REVIEWED_CURRENT_DIFFUSION_REVISION
