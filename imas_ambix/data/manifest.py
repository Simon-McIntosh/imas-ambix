"""FAIR-MAST shot manifest — read the parquet index, filter by camera coverage,
emit shot IDs for the bulk-download SLURM script.

The protocol is captured in ``plans/data-acquisition.md`` §4.2: the SLURM
script reads shot IDs from stdin and ``s5cmd cp``s each in turn. This
module is the source of those IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from imas_ambix.data.paths import SHOT_INDEX_LOCAL, SHOT_INDEX_URL


@dataclass(frozen=True)
class ShotManifest:
    """Filtered shot list with provenance metadata."""

    shot_ids: tuple[int, ...]
    source_url: str
    fetched_at: str
    filter_description: str
    total_in_index: int
    camera_flag_columns: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> str:
        return json.dumps(
            {
                "shot_ids": list(self.shot_ids),
                "source_url": self.source_url,
                "fetched_at": self.fetched_at,
                "filter_description": self.filter_description,
                "total_in_index": self.total_in_index,
                "camera_flag_columns": list(self.camera_flag_columns),
            },
            indent=2,
        )


def load_index(local: Path = SHOT_INDEX_LOCAL) -> pd.DataFrame:
    """Load the level-2 shot index from local parquet, or fetch if missing.

    The local copy is preferred because it captures a snapshot at mirror
    time. We fall back to fetching from mastapp.site so the probe can
    run before the mirror exists.
    """
    import pandas as pd  # lazy import: pandas is heavy at module-load

    if local.is_file():
        return pd.read_parquet(local)
    return pd.read_parquet(SHOT_INDEX_URL)


def detect_camera_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Return the index-column names that signal camera-group presence.

    FAIR-MAST encodes per-diagnostic availability as a column on the
    shot index. The exact column name depends on the index version; we
    pattern-match any column containing ``camera_visible`` or
    ``camera_ir``.
    """
    return tuple(c for c in df.columns if "camera_visible" in c or "camera_ir" in c)


def filter_camera_bearing(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of rows that carry visible or IR camera data."""
    cols = detect_camera_columns(df)
    if not cols:
        # No camera-flag columns found — return empty rather than guess.
        return df.iloc[0:0]
    mask = df[list(cols)].fillna(False).astype(bool).any(axis=1)
    return df[mask]


def build_manifest(
    df: pd.DataFrame,
    camera_only: bool,
) -> ShotManifest:
    """Build a :class:`ShotManifest` from the loaded index frame."""
    total = len(df)
    cols = detect_camera_columns(df)
    if camera_only:
        subset = filter_camera_bearing(df)
        desc = "camera-bearing shots (camera_visible or camera_ir column truthy)"
    else:
        subset = df
        desc = "all level-2 shots"
    ids = tuple(int(s) for s in subset["shot_id"].tolist())
    return ShotManifest(
        shot_ids=ids,
        source_url=SHOT_INDEX_URL,
        fetched_at=datetime.now(UTC).isoformat(),
        filter_description=desc,
        total_in_index=total,
        camera_flag_columns=cols,
    )


def emit_shot_ids(manifest: ShotManifest) -> str:
    """Render the shot IDs as a newline-separated string for stdin piping."""
    return "\n".join(str(s) for s in manifest.shot_ids) + "\n"
