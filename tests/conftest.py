"""Fixtures shared by the agent test suite.

The router gate's control file now defaults to the group-owned control directory
under the project base, ``<base_dir>/agents/control/router-gate.json``. That is
the right production default and the wrong thing for a test to resolve: it is a
real path on shared storage, so a test that resolved a control file with no
override would read and write production control state.

The fixture below therefore drives the site control path from its own
environment variable for every test. It sets that variable to the **empty
string** rather than to a scratch file, and the difference matters. An empty
site value suppresses the site branch, leaving a test that names nothing on the
lane document's sibling -- which is the contract the resolution tests pin and
which a scratch-file value would displace, sending those tests to a path that is
neither the sibling nor the production file. A test that is about the site
default sets the variable itself; either way no test can resolve the production
path.
"""

from __future__ import annotations

import pytest

from imas_ambix.agent.profile import GATE_CONTROL_PATH_ENV


@pytest.fixture(autouse=True)
def _no_production_gate_control_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test off the production gate control file."""
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, "")
