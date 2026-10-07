"""The single owner of the JT-60SA bundle-present decision for the test suite.

The JT-60SA maps are private, so every census test that reads them must decide
whether a bundle is reachable and skip when it is not.  That decision used to be
re-made in each test module by catching every :class:`MachineMapError` and
skipping, which reads a *broken* bundle as an absent one: a descriptor the
loader cannot parse, or a machine two bundles both carry, made the module skip
and the census passed quietly.

Presence is therefore decided here, once, from :func:`discover_bundles` alone.  A
directory named by ``IMAS_ALAMBIC_MAP_PATH`` is expected to carry a
``bundle.json``; a search-path entry with no descriptor at all contributes no
bundle, so a bare directory reads as absent and the honest skip applies.  Every
other outcome raises :class:`MachineMapError` while this module is imported, so
the importing module fails to collect instead of skipping:

* a descriptor that is present but unreadable or malformed, and
* a machine carried by more than one reachable bundle.

``BUNDLE`` is the one reachable bundle carrying jt-60sa, or ``None`` when none
does.  ``requires_jt60sa`` is the skip marker built from it; a module that also
needs the converted description store combines its own store check with
``SKIP_REASON``.
"""

from __future__ import annotations

import pytest

from imas_alambic.machine_map import (
    MachineMapError,
    MapBundle,
    discover_bundles,
)

#: The machine whose private bundle the census suites read.
MACHINE = "jt-60sa"

#: The reason a census module skips when no reachable bundle carries jt-60sa.
SKIP_REASON = (
    "no reachable map bundle carries 'jt-60sa'; set IMAS_ALAMBIC_MAP_PATH"
)

#: The reason a module skips when no bundle at all is reachable.
DISCOVERY_SKIP_REASON = "no reachable map bundle; set IMAS_ALAMBIC_MAP_PATH"


def reachable_bundles() -> tuple[MapBundle, ...]:
    """Return every reachable bundle, treating a descriptorless entry as none.

    A search-path directory that carries no ``bundle.json`` names no bundle, so
    it is dropped and discovery continues as if it were not there.  A descriptor
    that exists but cannot be read raises, and so does any other refusal, so a
    broken bundle is reported rather than read as absent.
    """

    try:
        return discover_bundles()
    except MachineMapError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return ()
        raise


def jt60sa_bundle() -> MapBundle | None:
    """Return the one reachable bundle carrying jt-60sa, or ``None`` if none.

    A machine carried by two distinct bundles raises, naming both, rather than
    one silently shadowing the other.
    """

    carried = tuple(
        bundle for bundle in reachable_bundles() if MACHINE in bundle.machines
    )
    if len(carried) > 1:
        named = ", ".join(f"{bundle.name!r} ({bundle.root})" for bundle in carried)
        raise MachineMapError(
            f"machine {MACHINE!r} is carried by more than one bundle: {named}"
        )
    return carried[0] if carried else None


#: The one reachable bundle carrying jt-60sa, or ``None``.  Raises on a broken
#: bundle while this module is imported, so the importing module fails to collect.
BUNDLE = jt60sa_bundle()

#: Skip a census module when no reachable bundle carries jt-60sa.
requires_jt60sa = pytest.mark.skipif(BUNDLE is None, reason=SKIP_REASON)

#: Skip when no bundle at all is reachable, whatever machine a module is about.
requires_bundle_discovery = pytest.mark.skipif(
    not reachable_bundles(), reason=DISCOVERY_SKIP_REASON
)
