"""Canonical paths for the FAIR-MAST mirror and probe artefacts.

These constants are the single source of truth — every other module
imports from here so we don't accidentally hardcode the mirror layout in
multiple places. The values match ``plans/data-acquisition.md`` §4.1.
"""

from __future__ import annotations

from pathlib import Path

# --- Endpoint ----------------------------------------------------------

S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
S3_BUCKET = "mast"
SHOT_INDEX_URL = "https://mastapp.site/parquet/level2/shots"
REST_API_BASE = "https://mastapp.site/json"

# --- Local mirror layout ----------------------------------------------

MIRROR_ROOT = Path("/work/projects/imas_gpu/mast")
LEVEL2_DIR = MIRROR_ROOT / "level2" / "shots"
MANIFEST_DIR = MIRROR_ROOT / "manifests"
PROBE_DIR = MIRROR_ROOT / ".probe"
SHOT_INDEX_LOCAL = MIRROR_ROOT / "shots-index.parquet"

# --- Tokens (separate root) -------------------------------------------

TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")
CHECKPOINT_ROOT = Path("/work/projects/imas_gpu/mast-checkpoints")


def s3_shot_path(shot_id: int) -> str:
    """Return the ``s3://`` URI for a level-2 shot's Zarr root."""
    return f"s3://{S3_BUCKET}/level2/shots/{shot_id}.zarr"


def local_shot_path(shot_id: int) -> Path:
    """Return the local path for a level-2 shot's Zarr root after mirror."""
    return LEVEL2_DIR / f"{shot_id}.zarr"
