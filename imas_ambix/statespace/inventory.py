"""Per-shot family inventory for FAIR-MAST level-1 Zarr corpus.

Builds a shot x family co-availability matrix by listing top-level
groups in each shot's Zarr directory (pure filesystem operations — no
Zarr open, no network I/O).

Usage
-----
    from imas_ambix.statespace.inventory import build_inventory, InventoryResult
    result = build_inventory(max_workers=16)
    result.save(Path("statespace/artifacts/family_inventory.json"))
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known group classifications
# ---------------------------------------------------------------------------

# Groups NOT in LEVEL1_SOURCES that appear in the wild — classified here so
# the leakage audit can scan them properly.
EXTRA_GROUP_CLASSIFICATION: dict[str, str] = {
    "ada": "dalpha_analysis",  # processed Dα: dalpha_integrated, dalpha_inverted, …
    "adg": "density_gradient_analysis",  # density_gradient, gradient_position — NOT Dα
    "aim": "dalpha_filterscope_analysis",  # da_hm10_t, da_to10 analysed subset
    "air": "ir_analysis",  # IR camera heat-load analysis (air = infra-red)
    "aoe": "microwave_reflectometry",  # co2_frac, ka_band, k_band, … — NOT Dα
    "asx": "soft_xray_sawtooth",  # elm_freqs, sawtooth detection — NOT Dα
    "esx": "equilibrium_sawtooth",  # lower/upper inversion radii — NOT Dα
    "rca": "camera_visible_rca",  # 2-D visible camera image (no time axis) — NOT Dα
    "xma": "magnetics_raw_a",  # raw magnetics group A
    "xmb": "magnetics_raw_b",
    "xmc": "magnetics_raw_c",
    "xmo": "magnetics_omaha",  # high-rate (MHz) Omaha magnetics raw
}

# Groups excluded from input families (solvers/reconstructions and controls)
EXCLUDED_GROUPS: frozenset[str] = frozenset({"efm", "esm", "xdc"})

# Groups that are raw measured magnetics — used to track the 'magnetics' family
MAGNETICS_GROUPS: frozenset[str] = frozenset(
    {"ama", "amb", "amc", "amh", "amm", "asm", "xma", "xmb", "xmc", "xmo"}
)

# Groups containing Dα channels (used for leakage audit)
DALPHA_GROUPS: frozenset[str] = frozenset({"xim", "ada", "aim"})

# Camera groups (visible + IR frame data)
CAMERA_GROUPS: frozenset[str] = frozenset(
    {"rba", "rbb", "rbc", "rco", "rgb", "rgc", "rir", "rit", "rzz", "rca"}
)


def _list_shot_groups(shot_zarr_path: Path) -> tuple[int, tuple[str, ...]]:
    """Return (shot_id, group_names) by listing the shot's Zarr directory.

    Pure filesystem operation — no Zarr/HDF5 open. Groups are the immediate
    subdirectories of the shot's Zarr root.
    """
    shot_id = int(shot_zarr_path.stem)
    try:
        groups = tuple(sorted(p.name for p in shot_zarr_path.iterdir() if p.is_dir()))
    except OSError as e:
        logger.warning("Cannot list %s: %s", shot_zarr_path, e)
        groups = ()
    return shot_id, groups


@dataclass
class InventoryResult:
    """Full per-shot group inventory + derived co-availability metrics.

    Attributes
    ----------
    shot_groups:
        Mapping from shot_id to the sorted tuple of top-level Zarr groups
        present for that shot.
    all_groups:
        Union of all group names seen across the corpus (sorted).
    n_shots:
        Total number of shots in the inventory.
    """

    shot_groups: dict[int, tuple[str, ...]] = field(default_factory=dict)
    all_groups: list[str] = field(default_factory=list)
    n_shots: int = 0

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    def shots_with_group(self, group: str) -> list[int]:
        """Return all shot IDs that carry *group*."""
        return [sid for sid, grps in self.shot_groups.items() if group in grps]

    def shots_with_all_groups(self, *groups: str) -> list[int]:
        """Return shots that carry ALL of the listed groups."""
        group_set = frozenset(groups)
        return [
            sid for sid, grps in self.shot_groups.items() if group_set.issubset(grps)
        ]

    def group_coverage(self) -> dict[str, int]:
        """Return {group: n_shots_present} for every group in the corpus."""
        counts: dict[str, int] = {}
        for grps in self.shot_groups.values():
            for g in grps:
                counts[g] = counts.get(g, 0) + 1
        return dict(sorted(counts.items()))

    def coavailability_matrix(self) -> np.ndarray:
        """Boolean array of shape (n_shots, n_groups) in group-sorted order."""
        shots = sorted(self.shot_groups.keys())
        groups = self.all_groups
        mat = np.zeros((len(shots), len(groups)), dtype=bool)
        g_idx = {g: i for i, g in enumerate(groups)}
        for s_idx, sid in enumerate(shots):
            for g in self.shot_groups[sid]:
                if g in g_idx:
                    mat[s_idx, g_idx[g]] = True
        return mat

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        coverage = self.group_coverage()
        return {
            "n_shots": self.n_shots,
            "all_groups": self.all_groups,
            "group_coverage": coverage,
            # Compact encoding: list of [shot_id, [groups...]] pairs
            "shot_groups": [
                [sid, list(grps)] for sid, grps in sorted(self.shot_groups.items())
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8"
        )
        logger.info("Inventory saved to %s (%d shots)", path, self.n_shots)

    @classmethod
    def load(cls, path: Path) -> InventoryResult:
        d = json.loads(path.read_text(encoding="utf-8"))
        shot_groups = {int(sid): tuple(grps) for sid, grps in d["shot_groups"]}
        return cls(
            shot_groups=shot_groups,
            all_groups=d["all_groups"],
            n_shots=d["n_shots"],
        )


def build_inventory(
    level1_dir: Path | None = None,
    max_workers: int = 8,
) -> InventoryResult:
    """Scan all shots in *level1_dir* and return an :class:`InventoryResult`.

    Parameters
    ----------
    level1_dir:
        Root directory containing ``<shot_id>.zarr`` subdirectories.
        Defaults to :data:`~imas_ambix.data.paths.LEVEL1_DIR`.
    max_workers:
        Number of worker processes for parallel directory listing.
        Each listing is a cheap ``os.listdir`` — no I/O beyond metadata.

    Returns
    -------
    InventoryResult
        Populated inventory covering all shots found in *level1_dir*.
    """
    root = Path(level1_dir) if level1_dir else LEVEL1_DIR
    shot_paths = sorted(root.glob("*.zarr"), key=lambda p: int(p.stem))
    logger.info(
        "Scanning %d shots in %s with %d workers", len(shot_paths), root, max_workers
    )

    shot_groups: dict[int, tuple[str, ...]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_list_shot_groups, p): p for p in shot_paths}
        for n_done, fut in enumerate(as_completed(futures), start=1):
            shot_id, groups = fut.result()
            shot_groups[shot_id] = groups
            if n_done % 2000 == 0:
                logger.info("  … %d / %d shots scanned", n_done, len(shot_paths))

    all_groups = sorted({g for grps in shot_groups.values() for g in grps})

    return InventoryResult(
        shot_groups=shot_groups,
        all_groups=all_groups,
        n_shots=len(shot_groups),
    )
