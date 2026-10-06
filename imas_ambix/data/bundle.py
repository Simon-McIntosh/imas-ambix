"""Ambix's map bundle, exposed to imas-alambic through its entry-point group.

The engine (imas-alambic) ships no maps of its own; it discovers bundles.  This
module is ambix's registration target: it names the directory that holds
ambix's ``bundle.json``, ``machine_maps/`` and ``maps/``.  The on-disk layout is
still owned by :mod:`imas_ambix.data.paths`; this module only points the engine
at it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imas_ambix.data import paths as _paths

if TYPE_CHECKING:
    from pathlib import Path


def load_bundle() -> Path:
    """Return ambix's bundle directory for the ``imas_alambic.bundles`` group."""

    return _paths.BUNDLE_ROOT
