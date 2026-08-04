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
from nova.transport import CurrentDiffusion as NovaCurrentDiffusion

from imas_ambix import physics

NOVA_REVISION = "30a0e25e507b1d025e40e914a7d468d3bca07a3c"
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


def test_installed_nova_distribution_comes_from_the_pinned_commit():
    direct_url = json.loads(distribution("nova-stella").read_text("direct_url.json"))
    assert direct_url["vcs_info"]["commit_id"] == NOVA_REVISION
