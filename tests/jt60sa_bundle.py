"""The single owner of the JT-60SA bundle-present decision for the test suite.

The JT-60SA maps are private, so every census test that reads them must decide
whether a bundle is reachable and skip when it is not.  That decision used to be
re-made in each test module by catching every :class:`MachineMapError` and
skipping, which reads a *broken* bundle as an absent one: a descriptor the
loader cannot parse, or a machine two bundles both carry, made the module skip
and the census passed quietly.

Presence is therefore decided here, once, from
:func:`imas_alambic.machine_map.bundles_carrying`.  A directory named by
``IMAS_ALAMBIC_MAP_PATH`` is expected to carry a ``bundle.json``; a descriptor
that is present but unreadable or malformed raises while this module is
imported, so the importing module fails to collect instead of skipping, and an
empty directory on the map path is exactly that case.  A machine carried by
more than one reachable bundle is likewise refused, through the single
:func:`imas_alambic.machine_map.bundle_for_machine` policy rather than a second
copy of its message.

``BUNDLE`` is the one reachable bundle carrying jt-60sa, or ``None`` when none
carries it.  ``requires_jt60sa`` is the skip marker built from it; a module that
also needs the converted description store combines its own store check with
``SKIP_REASON``.  ``requires_mast`` is the marker the public MAST-guarded tests
use, so those run whenever a bundle carries mast, whether or not the private
JT-60SA bundle is present.
"""

from __future__ import annotations

import pytest

from imas_alambic.machine_map import (
    MapBundle,
    bundle_for_machine,
    bundles_carrying,
    discover_bundles,
)

#: The machine whose private bundle the census suites read.
MACHINE = "jt-60sa"

#: The public machine ambix's own bundle carries; the MAST-guarded tests key on
#: it so they do not skip merely because the private JT-60SA bundle is absent.
MAST_MACHINE = "mast"

#: The reason a census module skips when no reachable bundle carries jt-60sa.
SKIP_REASON = (
    "no reachable map bundle carries 'jt-60sa'; set IMAS_ALAMBIC_MAP_PATH"
)

#: The reason a MAST-guarded module skips when no reachable bundle carries mast.
MAST_SKIP_REASON = "no reachable map bundle carries 'mast'"


def carries_machine(machine: str) -> bool:
    """Whether any reachable bundle declares ``machine``.

    The one predicate the MAST-guarded tests share, so they key on the machine
    they actually read rather than on whether any bundle at all is reachable.
    """

    return bool(bundles_carrying(machine))


def reachable_bundles() -> tuple[MapBundle, ...]:
    """Return every reachable bundle.

    A search-path directory that carries no ``bundle.json`` raises from the
    loader rather than contributing nothing, so an empty directory named by
    ``IMAS_ALAMBIC_MAP_PATH`` fails collection as a descriptor refusal, and is
    never read as an absent bundle.
    """

    return discover_bundles()


def jt60sa_bundle() -> MapBundle | None:
    """Return the one reachable bundle carrying jt-60sa, or ``None`` if none.

    An empty result is absence; any other outcome is delegated to
    :func:`bundle_for_machine`, whose policy refuses a machine carried by two
    distinct bundles naming both, rather than one silently shadowing the other.
    """

    if not bundles_carrying(MACHINE):
        return None
    return bundle_for_machine(MACHINE)


#: The one reachable bundle carrying jt-60sa, or ``None``.  Raises on a broken
#: bundle while this module is imported, so the importing module fails to collect.
BUNDLE = jt60sa_bundle()

#: Skip a census module when no reachable bundle carries jt-60sa.
requires_jt60sa = pytest.mark.skipif(BUNDLE is None, reason=SKIP_REASON)

#: Skip a public MAST-guarded module when no reachable bundle carries mast.
requires_mast = pytest.mark.skipif(
    not carries_machine(MAST_MACHINE), reason=MAST_SKIP_REASON
)
