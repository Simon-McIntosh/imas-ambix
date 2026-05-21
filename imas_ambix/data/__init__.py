"""FAIR-MAST data acquisition and access.

See ``plans/data-acquisition.md`` for the protocol that this module
implements: endpoint inventory, sizing-probe protocol, bulk-download
SLURM spec, and storage layout under
``/work/projects/imas_gpu/mast/``.
"""

from __future__ import annotations

from imas_ambix.data.paths import MIRROR_ROOT, PROBE_DIR, SHOT_INDEX_URL

__all__ = ["MIRROR_ROOT", "PROBE_DIR", "SHOT_INDEX_URL"]
