"""Dependency and ownership checks for the Nova compatibility facade."""

import json
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

NOVA_REVISION = "9c18d31f8bf0f424bd94d0a5c4ce5f1622d9fd8f"
ROOT = Path(__file__).parents[2]


def test_project_and_lock_pin_the_same_nova_revision():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["requires-python"] == ">=3.14"
    dependency = (
        f"nova-stella @ git+https://github.com/Simon-McIntosh/nova.git@{NOVA_REVISION}"
    )
    assert dependency in project["project"]["dependencies"]

    lock = (ROOT / "uv.lock").read_text()
    assert 'requires-python = ">=3.14"' in lock
    assert (
        "https://github.com/Simon-McIntosh/nova.git?"
        f"rev={NOVA_REVISION}#{NOVA_REVISION}"
    ) in lock


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


def test_installed_nova_distribution_comes_from_the_pinned_commit():
    direct_url = json.loads(distribution("nova-stella").read_text("direct_url.json"))
    assert direct_url["vcs_info"]["commit_id"] == NOVA_REVISION
