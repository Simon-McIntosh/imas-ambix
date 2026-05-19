"""Canonical paths for the FAIR-MAST mirror and probe artefacts.

These constants are the single source of truth — every other module
imports from here so we don't accidentally hardcode the mirror layout in
multiple places. The values match ``plans/data-acquisition.md`` §4.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Tier = Literal["level1", "level2"]

# --- Endpoint ----------------------------------------------------------

S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
S3_BUCKET = "mast"
SHOT_INDEX_URL = "https://mastapp.site/parquet/level2/shots"
REST_API_BASE = "https://mastapp.site/json"

# --- Local mirror layout ----------------------------------------------

MIRROR_ROOT = Path("/work/projects/imas_gpu/mast")
LEVEL1_DIR = MIRROR_ROOT / "level1" / "shots"
LEVEL2_DIR = MIRROR_ROOT / "level2" / "shots"
MANIFEST_DIR = MIRROR_ROOT / "manifests"
PROBE_DIR = MIRROR_ROOT / ".probe"
SHOT_INDEX_LOCAL = MIRROR_ROOT / "shots-index.parquet"

# --- Tokens (separate root) -------------------------------------------

TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")
CHECKPOINT_ROOT = Path("/work/projects/imas_gpu/mast-checkpoints")

# --- Level-1 source → IMAS group mapping ------------------------------
#
# Verbatim from
# ``ukaea/fair-mast-ingestion/mappings/level1/mast/groups.json``. Note:
# these mappings describe the *intended* level-2 ingestion. As of the
# 2026-05-19 probe (see plans/data-acquisition.md §10), the camera
# sources have not been ingested into level-2 — they exist only at
# level-1. The mapping is preserved here so future code can join the two
# tiers.

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
