"""Ambix's map bundles, exposed to imas-alambic through its entry-point group.

The engine (imas-alambic) ships no maps of its own; it discovers bundles.  This
module is ambix's registration target: it names the directory that holds
ambix's own ``bundle.json``, ``machine_maps/`` and ``maps/``, and it names every
machine directory under the repository's gitignored ``maps/`` tree that carries
a ``bundle.json``.  Those roots are read from :mod:`imas_ambix.data.paths`,
which owns them, rather than duplicated in ``bundle.json``: a store root has
exactly one owner.

Publishing the map directories beside ambix's own bundle lets a checkout with
the map directories populated resolve them with no ``IMAS_ALAMBIC_MAP_PATH``:
the engine reaches the same discovery group either way.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from imas_alambic.machine_map import MapBundle, load_bundle_descriptor
from imas_ambix.data import paths as _paths

_STORE_ROOTS = {
    "LEVEL2_DIR": _paths.LEVEL2_DIR,
    "eddb_cache": _paths.JT60SA_ROOT,
    "ids_root": _paths.JT60SA_IDS_DIR,
}

#: The map directory's descriptor, so a machine directory is a bundle exactly
#: when it names itself with one.
_BUNDLE_DESCRIPTOR = "bundle.json"


def _ambix_bundle() -> MapBundle:
    """Return ambix's own bundle, its store roots attached from ``paths``.

    The bundle descriptor names ambix's directory, machines and version; the
    store roots are attached here from :mod:`imas_ambix.data.paths` so the two
    never drift.
    """

    bundle = load_bundle_descriptor(_paths.BUNDLE_ROOT)
    return replace(bundle, store_roots=MappingProxyType(dict(_STORE_ROOTS)))


def _machine_bundles() -> tuple[MapBundle, ...]:
    """Return one bundle per machine directory under ``maps/`` that declares one.

    A directory without a ``bundle.json`` is not a bundle and contributes
    nothing, so a clone whose ``maps/`` tree is empty or partial still loads.
    """

    root = _paths.MAPS_DIR
    if not root.is_dir():
        return ()
    return tuple(
        load_bundle_descriptor(child)
        for child in sorted(root.iterdir())
        if (child / _BUNDLE_DESCRIPTOR).is_file()
    )


def load_bundle() -> tuple[MapBundle, ...]:
    """Return the bundles ambix publishes to the engine's discovery group.

    The ``imas_alambic.bundles`` entry point names this symbol, so its name is
    fixed by ``pyproject.toml``.  It returns ambix's own bundle first, then one
    bundle per machine directory under :data:`imas_ambix.data.paths.MAPS_DIR`
    that carries a ``bundle.json``.
    """

    return (_ambix_bundle(), *_machine_bundles())
