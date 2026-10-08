"""Canonical paths for the FAIR-MAST mirror and probe artefacts.

These constants are the single source of truth for the local mirror layout.
Other modules import from here so path definitions do not diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Tier = Literal["level1", "level2"]

# --- Ambix's map bundle ------------------------------------------------
#
# The engine (imas-alambic) carries code and schema, never map data.  Ambix's
# authored maps live here as a *bundle*: this directory holds ``bundle.json``
# beside ``machine_maps/`` and ``maps/``, and ambix registers it under the
# ``imas_alambic.bundles`` entry-point group.  These constants are ambix's own
# address for that bundle, so ambix code that reaches the map files directly
# (spine_bench, the JT-60SA map tests) does not go through the engine.
BUNDLE_ROOT = Path(__file__).resolve().parent
PACKAGED_MACHINE_MAP_ROOT = BUNDLE_ROOT / "machine_maps"
PACKAGED_MAP_ROOT = BUNDLE_ROOT / "maps"

# --- Map development directory ----------------------------------------
#
# Private machine maps are not tracked in git.  They live in ``maps/<machine>/``
# at the repository root, which ``.gitignore`` excludes, and the map release CLI
# publishes them to GHCR.  Ambix's ``imas_alambic.bundles`` entry point offers
# every machine directory under here that carries a ``bundle.json`` to the
# engine, so a checkout with this directory populated needs no
# ``IMAS_ALAMBIC_MAP_PATH``.
REPO_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = REPO_ROOT / "maps"
JT60SA_MAP_DIR = MAPS_DIR / "jt-60sa"

# --- Endpoint ----------------------------------------------------------

S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
S3_BUCKET = "mast"
SHOT_INDEX_URL = "https://mastapp.site/parquet/level2/shots"
REST_API_BASE = "https://mastapp.site/json"

# --- Local mirror layout ----------------------------------------------

MIRROR_ROOT = Path("/work/projects/imas_gpu/mast")

# --- JT-60SA on-demand EDDB cache -------------------------------------
#
# The JT-60SA cache is filled on demand over ssh from the Naka analysis
# server and is NOT a copy of the EDDB: only the channels a map binds, for
# the shots a run asks for, land here as ``{shot}.zarr/{category}/{dname}``.
# It sits on GPFS beside the MAST mirror so every ITER session and worker
# reads the same bytes.
JT60SA_ROOT = Path("/work/projects/imas_gpu/jt60sa")
JT60SA_IDS_DIR = JT60SA_ROOT / "ids"

# The description stores' ``source/`` and ``superseded/`` inputs sit beside the
# converted store inside the machine directory, so the description directory is
# the machine directory itself (``JT60SA_DESCRIPTION_DIR / "source"`` is the
# SELENE deck source).  A clone that carries no map directory leaves this path
# unpopulated rather than reaching a second copy under /work.
JT60SA_DESCRIPTION_DIR = JT60SA_MAP_DIR
LEVEL1_DIR = MIRROR_ROOT / "level1" / "shots"
LEVEL2_DIR = MIRROR_ROOT / "level2" / "shots"
MANIFEST_DIR = MIRROR_ROOT / "manifests"
PROBE_DIR = MIRROR_ROOT / ".probe"
SHOT_INDEX_LOCAL = MIRROR_ROOT / "shots-index.parquet"

# --- Tokens (separate root) -------------------------------------------

TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")
CHECKPOINT_ROOT = Path("/work/projects/imas_gpu/mast-checkpoints")

# --- Targets (separate root — eval-only, never an input) --------------
#
# The world-model PREDICTION targets (the L2 equilibrium reconstruction +
# the reconstruction-derived globals) live under their own root, NOT under
# ``TOKEN_ROOT``.  Because ``mast-targets`` is not a child of ``mast-tokens``,
# no glob the input loader runs over ``TOKEN_ROOT/v2/...`` can ever reach a
# target — the physical separation is the first leakage wall.  See
# :mod:`imas_ambix.tokenizer.store_targets`.
TARGET_ROOT = Path("/work/projects/imas_gpu/mast-targets")

# --- Level-1 source → IMAS group mapping ------------------------------
#
# Verbatim from
# ``ukaea/fair-mast-ingestion/mappings/level1/mast/groups.json``. These
# mappings describe the intended level-2 ingestion. The camera sources exist
# only at level-1, so the mapping also records how to join the two tiers when
# their level-2 counterparts become available.

LEVEL1_SOURCES = {
    "rba": "camera_visible.camera_lower",
    "rbb": "camera_visible.camera_center",
    "rbc": "camera_visible.camera_lower_alt",
    "rco": "camera_visible.camera_color",
    "rgb": "camera_visible.bremsstrahlung_a",
    "rgc": "camera_visible.bremsstrahlung_b",
    "rir": "camera_ir.divertor",
    "rit": "camera_ir.target",
    "rzz": "camera_visible.bremsstrahlung_zebra",
    "ama": "magnetics_a",
    "amb": "magnetics_b",
    "amc": "magnetics_c",
    "amh": "magnetics_h",
    "amm": "magnetics_omaha_mhz",
    "asm": "magnetics_saddle",
    "ams": "mse",
    "anb": "nbi",
    "ane": "interferometer",
    "anu": "neutron_diagnostic",
    "abm": "bolometer",
    "act": "charge_exchange",
    "aga": "gas_injection",
    "ahx": "hard_x_rays",
    "ait": "camera_ir",
    "alp": "langmuir_probes",
    "atm": "thomson_scattering_core",
    "ayc": "thomson_scattering_combined",
    "aye": "thomson_scattering_edge",
    "efm": "equilibrium_efit",
    "esm": "equilibrium_solovev",
    "xdc": "pulse_schedule",
    "xim": "spectrometer_visible",
    "xsx": "soft_x_rays",
    "xma": "magnetics_raw_a",
    "xmb": "magnetics_raw_b",
    "xmc": "magnetics_raw_c",
    "xmo": "magnetics_omaha",
}

CAMERA_SOURCES = ("rba", "rbb", "rbc", "rco", "rgb", "rgc", "rir", "rit", "rzz")
"""Level-1 source names that carry camera frame data."""

CONTROL_SOURCES = ("anb", "aga", "efm", "xdc")
"""Minimum control-vector sources for the world-model condition stream."""


def s3_shot_path(shot_id: int, tier: Tier = "level2") -> str:
    """Return the ``s3://`` URI for a shot's Zarr root at the given tier."""
    return f"s3://{S3_BUCKET}/{tier}/shots/{shot_id}.zarr"


def s3_group_path(shot_id: int, group: str, tier: Tier = "level2") -> str:
    """Return the ``s3://`` URI for one group of one shot at the given tier."""
    return f"s3://{S3_BUCKET}/{tier}/shots/{shot_id}.zarr/{group}"


def local_shot_path(shot_id: int, tier: Tier = "level2") -> Path:
    """Return the local mirror path for a shot's Zarr root at the given tier."""
    root = LEVEL1_DIR if tier == "level1" else LEVEL2_DIR
    return root / f"{shot_id}.zarr"
