"""FAIR-MAST shot manifest — load the parquet index, list S3 groups,
emit per-tier per-group download targets.

The 2026-05-19 probe (see ``plans/data-acquisition.md`` §10) established
that the level-2 corpus carries no camera groups; the camera data lives
in level-1 sources (``rba``, ``rbb``, ``rir``, …). The parquet shot
index has no camera-presence flag columns. Group inventories are
therefore built from **S3 listings**, not the parquet index.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from imas_ambix.data.paths import (
    S3_BUCKET,
    S3_ENDPOINT,
    SHOT_INDEX_LOCAL,
    SHOT_INDEX_URL,
    Tier,
)

# --- Parquet index ----------------------------------------------------


def load_index(local: Path = SHOT_INDEX_LOCAL) -> pd.DataFrame:
    """Load the level-2 shot index from local parquet, or fetch from HTTP.

    Despite the level-2 name in the URL, the parquet index lists every
    shot known to FAIR-MAST regardless of tier; it carries metadata only
    (campaign, plasma quantities, timestamp). It carries **no** group-
    presence columns. Use :func:`list_groups_for_shot` to discover which
    groups a shot actually has in the bucket.
    """
    import pandas as pd  # lazy import: pandas is heavy at module-load

    if local.is_file():
        return pd.read_parquet(local)
    return pd.read_parquet(SHOT_INDEX_URL)


def shot_ids_from_index(df: pd.DataFrame) -> tuple[int, ...]:
    """Return the full shot-id tuple from the index dataframe."""
    return tuple(int(s) for s in df["shot_id"].tolist())


def s5cmd_du(s3_path: str) -> tuple[int, int]:
    """Run ``s5cmd du <s3_path>`` and parse ``"N bytes in M objects:..."``.

    Returns ``(bytes, objects)``. Raises if s5cmd exits non-zero or the
    output line doesn't match. Suitable for both shot-level
    ``s3://mast/level2/shots/30420.zarr/*`` and group-level
    ``s3://mast/level2/shots/30420.zarr/magnetics/*`` prefixes.
    """
    s5cmd = _require_s5cmd()
    cmd = [
        s5cmd,
        "--no-sign-request",
        "--endpoint-url",
        S3_ENDPOINT,
        "du",
        s3_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        if "no object found" in (proc.stderr or "").lower():
            return (0, 0)
        raise RuntimeError(f"s5cmd du failed for {s3_path!r}: {proc.stderr!r}")
    # Expected line: "112119162 bytes in 1134 objects: s3://mast/..."
    for line in proc.stdout.splitlines():
        m = re.match(r"^(\d+)\s+bytes\s+in\s+(\d+)\s+objects", line)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def shot_ids_from_bucket(tier: Tier = "level2") -> tuple[int, ...]:
    """List shot IDs by ``s5cmd ls`` against the bucket prefix.

    This is the authoritative count for a given tier. The level-1 bucket
    carries more shots than the level-2 parquet index (17k vs 11.5k as of
    2026-05-19), so this listing is required for any level-1 work.
    """
    s5cmd = _require_s5cmd()
    target = f"s3://{S3_BUCKET}/{tier}/shots/"
    cmd = [s5cmd, "--no-sign-request", "--endpoint-url", S3_ENDPOINT, "ls", target]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"s5cmd ls {target} failed: {proc.stderr!r}")
    shots: list[int] = []
    for line in proc.stdout.splitlines():
        if "DIR" not in line:
            continue
        token = line.split()[-1].rstrip("/")
        # Expect names like "30420.zarr"
        if token.endswith(".zarr"):
            try:
                shots.append(int(token.removesuffix(".zarr")))
            except ValueError:
                continue
    return tuple(sorted(shots))


# --- S3 listing -------------------------------------------------------


class S5cmdMissingError(RuntimeError):
    """Raised when the s5cmd binary is not on PATH."""


def _require_s5cmd() -> str:
    path = shutil.which("s5cmd")
    if path is None:
        raise S5cmdMissingError(
            "s5cmd is not on PATH — see plans/data-acquisition.md §3.1 for "
            "the install command"
        )
    return path


def list_groups_for_shot(shot_id: int, tier: Tier = "level2") -> tuple[str, ...]:
    """List the immediate group sub-directories of a shot in the S3 bucket.

    Runs ``s5cmd ls`` against the shot's Zarr root and parses the
    DIR-prefixed lines. Returns an empty tuple if the shot does not
    exist (``no object found``) — that's the bucket's way of saying
    the shot is unknown at this tier.
    """
    s5cmd = _require_s5cmd()
    target = f"s3://{S3_BUCKET}/{tier}/shots/{shot_id}.zarr/"
    cmd = [s5cmd, "--no-sign-request", "--endpoint-url", S3_ENDPOINT, "ls", target]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        # "no object found" is normal for shots absent from the tier.
        if "no object found" in (proc.stderr or "").lower():
            return ()
        raise RuntimeError(
            f"s5cmd ls failed for shot {shot_id} tier {tier!r}: {proc.stderr!r}"
        )
    groups: list[str] = []
    for line in proc.stdout.splitlines():
        # s5cmd v2.x DIR lines look like:
        #   "                                  DIR  camera_visible/"
        if "DIR" not in line:
            continue
        token = line.split()[-1]
        if token.endswith("/"):
            groups.append(token.rstrip("/"))
    return tuple(sorted(groups))


def inventory_groups(
    shot_ids: list[int],
    tier: Tier = "level2",
    max_workers: int = 8,
) -> dict[int, tuple[str, ...]]:
    """Build a ``{shot_id: (group, ...)}`` inventory in parallel.

    The S3 listings are independent so a small thread pool keeps the
    probe responsive on a sample of 50-100 shots.
    """
    inv: dict[int, tuple[str, ...]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(list_groups_for_shot, sid, tier): sid for sid in shot_ids
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                inv[sid] = fut.result()
            except Exception as exc:  # noqa: BLE001 — surface as empty + log later
                inv[sid] = ()
                # Keep the failure in the inventory key as an annotation we
                # can spot later — print to stderr so the CLI rendering
                # still works.
                print(f"warn: {sid} listing failed: {exc!r}")
    return inv


def filter_by_groups(
    inventory: dict[int, tuple[str, ...]],
    required: tuple[str, ...],
    mode: str = "any",
) -> list[int]:
    """Return shot ids whose group set intersects ``required``.

    ``mode='any'`` returns shots that have at least one of the required
    groups. ``mode='all'`` returns shots that have every required group.
    """
    if mode not in ("any", "all"):
        raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")
    req_set = set(required)
    out: list[int] = []
    for sid, groups in inventory.items():
        present = req_set & set(groups)
        if mode == "any" and present or mode == "all" and present == req_set:
            out.append(sid)
    return sorted(out)


def sum_sizes_from_bucket(
    shot_ids: list[int],
    tier: Tier = "level2",
    groups: tuple[str, ...] = (),
    max_workers: int = 16,
) -> dict[int, tuple[int, int]]:
    """Parallel `s5cmd du` over many shot or shot/group prefixes.

    Returns ``{shot_id: (bytes, objects)}``. ``s5cmd du`` reads from S3
    metadata without downloading payload bytes, so this is the right way
    to size a tier or sub-prefix in advance of the bulk download.
    """

    def _one(sid: int) -> tuple[int, tuple[int, int]]:
        base = f"s3://{S3_BUCKET}/{tier}/shots/{sid}.zarr"
        if not groups:
            return sid, s5cmd_du(f"{base}/*")
        total_b = 0
        total_o = 0
        for g in groups:
            b, o = s5cmd_du(f"{base}/{g}/*")
            total_b += b
            total_o += o
        return sid, (total_b, total_o)

    out: dict[int, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed({pool.submit(_one, sid) for sid in shot_ids}):
            try:
                sid, sz = fut.result()
                out[sid] = sz
            except Exception as exc:  # noqa: BLE001
                print(f"warn: du failed: {exc!r}")
    return out


def group_coverage(
    inventory: dict[int, tuple[str, ...]],
) -> dict[str, int]:
    """Return ``{group_name: number_of_shots_with_this_group}``."""
    counts: dict[str, int] = {}
    for groups in inventory.values():
        for g in groups:
            counts[g] = counts.get(g, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# --- Manifest objects -------------------------------------------------


@dataclass(frozen=True)
class DownloadTarget:
    """One ``s5cmd cp`` job: a shot's group at a tier."""

    shot_id: int
    tier: str  # "level1" or "level2"
    group: str | None  # None = whole shot


@dataclass(frozen=True)
class ShotManifest:
    """Tier-aware shot list with provenance metadata."""

    tier: str
    shot_ids: tuple[int, ...]
    groups: tuple[str, ...]
    """Selected groups; empty tuple means "all groups in each shot"."""
    fetched_at: str
    source: str
    filter_description: str
    total_in_index: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "tier": self.tier,
                "shot_ids": list(self.shot_ids),
                "groups": list(self.groups),
                "fetched_at": self.fetched_at,
                "source": self.source,
                "filter_description": self.filter_description,
                "total_in_index": self.total_in_index,
            },
            indent=2,
        )

    def targets(self) -> list[DownloadTarget]:
        """Cross-product of shots × groups, ready for `s5cmd cp` emission."""
        if not self.groups:
            return [
                DownloadTarget(shot_id=s, tier=self.tier, group=None)
                for s in self.shot_ids
            ]
        return [
            DownloadTarget(shot_id=s, tier=self.tier, group=g)
            for s in self.shot_ids
            for g in self.groups
        ]


def build_manifest(
    tier: Tier,
    shot_ids: list[int],
    groups: tuple[str, ...] = (),
    total_in_index: int = 0,
    filter_description: str = "",
) -> ShotManifest:
    """Build a tier-aware :class:`ShotManifest`."""
    return ShotManifest(
        tier=tier,
        shot_ids=tuple(int(s) for s in shot_ids),
        groups=tuple(groups),
        fetched_at=datetime.now(UTC).isoformat(),
        source=SHOT_INDEX_URL,
        filter_description=filter_description or f"{len(shot_ids)} shots at {tier}",
        total_in_index=total_in_index,
    )


def emit_shot_ids(manifest: ShotManifest) -> str:
    """Render manifest's shot IDs as a newline-separated string for stdin piping."""
    return "\n".join(str(s) for s in manifest.shot_ids) + "\n"


def emit_targets_as_s5cmd(manifest: ShotManifest) -> str:
    """Render the manifest as a stream of ``s5cmd cp`` lines.

    The output goes through ``s5cmd run`` for batched parallel execution
    (much faster than spawning one s5cmd per shot in a shell loop).
    """
    lines: list[str] = []
    for t in manifest.targets():
        if t.group is None:
            src = f"s3://{S3_BUCKET}/{t.tier}/shots/{t.shot_id}.zarr/*"
            dst = f"./{t.tier}/shots/{t.shot_id}.zarr/"
        else:
            src = f"s3://{S3_BUCKET}/{t.tier}/shots/{t.shot_id}.zarr/{t.group}/*"
            dst = f"./{t.tier}/shots/{t.shot_id}.zarr/{t.group}/"
        lines.append(f"cp {src} {dst}")
    return "\n".join(lines) + "\n"
