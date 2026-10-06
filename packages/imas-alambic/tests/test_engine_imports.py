"""Every engine module imports with only the engine's own dependencies present.

The facility server installs the engine into a clean environment, so an import
of any module that reaches for a package the engine does not declare is a
failure there rather than at some later call.  Each module is imported on its
own so a break names the module that carries it.
"""

from __future__ import annotations

import importlib

import pytest

ENGINE_MODULES = [
    "imas_alambic.cocos",
    "imas_alambic.machine_map",
    "imas_alambic.signal_map",
    "imas_alambic.transform_engine",
    "imas_alambic.virtual_zarr",
    "imas_alambic.eddb",
    "imas_alambic.eddb_remote",
    "imas_alambic.pulse_mint",
    "imas_alambic.cli",
]


@pytest.mark.parametrize("module", ENGINE_MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
