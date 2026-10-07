"""Ambix's map bundle, exposed to imas-alambic through its entry-point group.

The engine (imas-alambic) ships no maps of its own; it discovers bundles.  This
module is ambix's registration target: it names the directory that holds
ambix's ``bundle.json``, ``machine_maps/`` and ``maps/``, and it supplies the
named store roots the engine resolves against.  Those roots are read from
:mod:`imas_ambix.data.paths`, which owns them, rather than duplicated in
``bundle.json``: a store root has exactly one owner.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from imas_alambic.machine_map import MapBundle, load_bundle_descriptor
from imas_ambix.data import paths as _paths

_STORE_ROOTS = {
    "LEVEL2_DIR": _paths.LEVEL2_DIR,
    "JT60SA_ROOT": _paths.JT60SA_ROOT,
    "JT60SA_DESCRIPTION_DIR": _paths.JT60SA_DESCRIPTION_DIR,
    "eddb_cache": _paths.JT60SA_ROOT,
}


def load_bundle() -> MapBundle:
    """Return ambix's bundle for the ``imas_alambic.bundles`` group.

    The bundle descriptor names ambix's directory, machines and version; the
    store roots are attached here from :mod:`imas_ambix.data.paths` so the two
    never drift.
    """

    bundle = load_bundle_descriptor(_paths.BUNDLE_ROOT)
    return replace(bundle, store_roots=MappingProxyType(dict(_STORE_ROOTS)))
