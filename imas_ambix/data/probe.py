"""FAIR-MAST sizing + group-inventory probe.

Implements the protocol in ``plans/data-acquisition.md`` §3 and the
2026-05-19 findings in §10:

1. Pull the parquet shot index (metadata-only — no group flags).
2. Sample N shots and **list** the bucket prefix to learn which groups
   each shot actually carries at the chosen tier.
3. Copy a small subset to measure sustained throughput + per-shot size.
4. Report acceptance against the per-tier thresholds.

Run from any host that has both outbound network and (for caching to the
final mirror location) GPFS group access. The login node satisfies both;
the betelgeuse GPU node does not.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from imas_ambix.data.paths import (
    CAMERA_SOURCES,
    S3_BUCKET,
    S3_ENDPOINT,
    Tier,
)

# Acceptance thresholds — see plans/data-acquisition.md §3.2 + §10.6
# Lowered after the 2026-05-19 probe found the real corpus is ~1.5 TB.
THROUGHPUT_MIN_MBPS = 200.0
THROUGHPUT_DEGRADED_MBPS = 50.0
SHOT_P95_MAX_GB = 5.0
TOTAL_SIZE_MIN_TB = 0.05  # 50 GB — sanity check, not an upper bound on the corpus
TOTAL_SIZE_MAX_TB = 12.0
CAMERA_COVERAGE_MIN = 0.3  # fraction of sampled shots that carry any camera source


@dataclass
class ShotSample:
    """Per-shot result from the probe."""

    shot_id: int
    groups: tuple[str, ...]
    bytes_copied: int
    elapsed_s: float
    error: str | None = None

    @property
    def has_camera(self) -> bool:
        return bool(
            set(self.groups) & set(CAMERA_SOURCES + ("camera_visible", "camera_ir"))
        )

    @property
    def throughput_mbps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return (self.bytes_copied / 1e6) / self.elapsed_s


@dataclass
class ProbeReport:
    """Aggregate probe result, serialisable to JSON."""

    tier: str
    n_shots_in_index: int
    n_shots_in_tier: int
    sample_size: int
    samples: list[ShotSample] = field(default_factory=list)
    sustained_throughput_mbps: float = 0.0
    median_shot_size_mb: float = 0.0
    p95_shot_size_mb: float = 0.0
    extrapolated_total_size_tb: float = 0.0
    group_coverage: dict[str, int] = field(default_factory=dict)
    camera_coverage_fraction: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "tier": self.tier,
            "n_shots_in_index": self.n_shots_in_index,
            "n_shots_in_tier": self.n_shots_in_tier,
            "sample_size": self.sample_size,
            "sustained_throughput_mbps": self.sustained_throughput_mbps,
            "median_shot_size_mb": self.median_shot_size_mb,
            "p95_shot_size_mb": self.p95_shot_size_mb,
            "extrapolated_total_size_tb": self.extrapolated_total_size_tb,
            "group_coverage": self.group_coverage,
            "camera_coverage_fraction": self.camera_coverage_fraction,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": self.notes,
            "acceptance": self.acceptance_summary(),
            "samples": [
                {
                    "shot_id": s.shot_id,
                    "groups": list(s.groups),
                    "has_camera": s.has_camera,
                    "bytes_copied": s.bytes_copied,
                    "elapsed_s": s.elapsed_s,
                    "throughput_mbps": s.throughput_mbps,
                    "error": s.error,
                }
                for s in self.samples
            ],
        }
        return json.dumps(payload, indent=2)

    def acceptance_summary(self) -> dict[str, str]:
        """Map each gate to ``"pass"``, ``"degraded"``, or ``"fail"``."""
        out: dict[str, str] = {}
        if self.sustained_throughput_mbps >= THROUGHPUT_MIN_MBPS:
            out["throughput"] = "pass"
        elif self.sustained_throughput_mbps >= THROUGHPUT_DEGRADED_MBPS:
            out["throughput"] = "degraded"
        else:
            out["throughput"] = "fail"

        if TOTAL_SIZE_MIN_TB <= self.extrapolated_total_size_tb <= TOTAL_SIZE_MAX_TB:
            out["total_size"] = "pass"
        else:
            out["total_size"] = "fail"

        out["per_shot_p95"] = (
            "pass" if self.p95_shot_size_mb <= SHOT_P95_MAX_GB * 1024 else "fail"
        )

        # Camera coverage gate only meaningful at level-1, where cameras live.
        if self.tier == "level1":
            out["camera_coverage"] = (
                "pass"
                if self.camera_coverage_fraction >= CAMERA_COVERAGE_MIN
                else "fail"
            )
        else:
            out["camera_coverage"] = "n/a"
        return out


def s5cmd_available() -> bool:
    """Return True if the ``s5cmd`` binary is on PATH."""
    return shutil.which("s5cmd") is not None


def run_shot_copy(
    shot_id: int,
    dest_root: Path,
    tier: Tier = "level2",
    groups: tuple[str, ...] = (),
    numworkers: int = 32,
    timeout_s: float = 600.0,
) -> ShotSample:
    """Copy one shot (optionally a group subset) via ``s5cmd`` and time it.

    Bytes are measured by ``du``-ing the destination after copy. Errors
    from ``s5cmd`` are captured on the sample, not raised.

    The ``groups`` attribute on the returned :class:`ShotSample` reflects
    *what was actually copied* — empty if every requested group was
    missing from the shot, populated otherwise. Callers that pass an
    empty ``groups`` (whole-shot copy) get back an empty tuple too; in
    that case the bytes-copied field is the source of truth.
    """
    dest = dest_root / f"{shot_id}.zarr"
    dest.mkdir(parents=True, exist_ok=True)
    base = f"s3://{S3_BUCKET}/{tier}/shots/{shot_id}.zarr"

    if groups:
        # Multi-group copy via s5cmd run (parallel within one process)
        run_lines = [f"cp {base}/{g}/* {dest}/{g}/" for g in groups]
        run_input = "\n".join(run_lines) + "\n"
        cmd = [
            "s5cmd",
            "--no-sign-request",
            "--endpoint-url",
            S3_ENDPOINT,
            "--numworkers",
            str(numworkers),
            "run",
        ]
        t0 = time.monotonic()
        error: str | None = None
        try:
            subprocess.run(
                cmd,
                input=run_input,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or "").strip()[:512] or "s5cmd nonzero exit"
        except subprocess.TimeoutExpired:
            error = f"timeout after {timeout_s:.0f}s"
        elapsed = time.monotonic() - t0
    else:
        # Whole-shot copy
        cmd = [
            "s5cmd",
            "--no-sign-request",
            "--endpoint-url",
            S3_ENDPOINT,
            "--numworkers",
            str(numworkers),
            "cp",
            f"{base}/*",
            str(dest) + "/",
        ]
        t0 = time.monotonic()
        error = None
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or "").strip()[:512] or "s5cmd nonzero exit"
        except subprocess.TimeoutExpired:
            error = f"timeout after {timeout_s:.0f}s"
        elapsed = time.monotonic() - t0

    bytes_copied = _du_bytes(dest)
    # Report what's actually on disk, group-wise — directories whose copy
    # returned no bytes are dropped so `has_camera` doesn't lie.
    actual_groups: tuple[str, ...] = ()
    if dest.exists():
        actual_groups = tuple(
            sorted(p.name for p in dest.iterdir() if p.is_dir() and _du_bytes(p) > 0)
        )
    return ShotSample(
        shot_id=shot_id,
        groups=actual_groups,
        bytes_copied=bytes_copied,
        elapsed_s=elapsed,
        error=error,
    )


def _du_bytes(path: Path) -> int:
    """Recursive sum of file sizes under ``path``, in bytes."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((len(s) - 1) * q))
    return s[k]


def sample_shots(
    df: pd.DataFrame,
    sample_size: int,
    seed: int = 0,
) -> list[int]:
    """Return a deterministic shot-id sample from the index."""
    if len(df) == 0:
        return []
    n = min(sample_size, len(df))
    return [int(s) for s in df.sample(n=n, random_state=seed)["shot_id"].tolist()]


def run_probe(
    sample_size: int = 50,
    tier: Tier = "level2",
    groups: tuple[str, ...] = (),
    numworkers: int = 32,
    timeout_s: float = 600.0,
    output_dir: Path | None = None,
    seed: int = 0,
    n_shots_in_tier: int | None = None,
) -> ProbeReport:
    """Run the full probe end-to-end and return the populated report.

    Heavy lifting (pandas import, ``s5cmd`` invocation, ``du``) lives
    inside this function so that importing the module is cheap.
    """
    from imas_ambix.data.manifest import (
        S5cmdMissingError,
        group_coverage,
        inventory_groups,
        load_index,
    )

    if not s5cmd_available():
        raise S5cmdMissingError(
            "s5cmd is not on PATH — install from "
            "https://github.com/peak/s5cmd/releases and put on PATH"
        )

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    df = load_index()
    n_index = len(df)

    sample_ids = sample_shots(df, sample_size, seed=seed)

    # Step 1: cheap S3 group inventory (parallel `s5cmd ls`)
    inv = inventory_groups(sample_ids, tier=tier, max_workers=8)
    coverage = group_coverage(inv)
    n_with_camera = sum(
        1
        for sid in inv
        if set(inv[sid]) & set(CAMERA_SOURCES + ("camera_visible", "camera_ir"))
    )
    camera_fraction = n_with_camera / max(len(inv), 1)

    samples: list[ShotSample] = []

    with tempfile.TemporaryDirectory(prefix="mast-probe-") as tmp:
        tmp_path = Path(tmp)
        for shot_id in sample_ids:
            present = set(inv.get(shot_id, ()))
            if not present:
                # Shot doesn't exist at this tier — record honestly.
                samples.append(
                    ShotSample(
                        shot_id=shot_id,
                        groups=(),
                        bytes_copied=0,
                        elapsed_s=0.0,
                        error="absent at tier",
                    )
                )
                continue
            if groups:
                effective = tuple(g for g in groups if g in present)
                if not effective:
                    # None of the requested groups are present — skip the
                    # copy but record so the report shows reality.
                    samples.append(
                        ShotSample(
                            shot_id=shot_id,
                            groups=(),
                            bytes_copied=0,
                            elapsed_s=0.0,
                            error="requested groups absent",
                        )
                    )
                    continue
            else:
                effective = ()
            s = run_shot_copy(
                shot_id,
                tmp_path,
                tier=tier,
                groups=effective,
                numworkers=numworkers,
                timeout_s=timeout_s,
            )
            samples.append(s)

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    successful = [s for s in samples if s.error is None and s.bytes_copied > 0]
    total_bytes = sum(s.bytes_copied for s in successful)
    total_seconds = sum(s.elapsed_s for s in successful)
    sustained_throughput_mbps = (
        (total_bytes / 1e6) / total_seconds if total_seconds > 0 else 0.0
    )

    sizes_mb = [s.bytes_copied / 1e6 for s in successful]
    median_mb = _percentile(sizes_mb, 0.5)
    p95_mb = _percentile(sizes_mb, 0.95)
    mean_mb = sum(sizes_mb) / len(sizes_mb) if sizes_mb else 0.0

    # Extrapolate to the entire tier; the caller may pass an authoritative
    # tier count (e.g. from `s5cmd ls bucket/level1/shots/ | wc -l`).
    n_for_extrap = n_shots_in_tier if n_shots_in_tier else n_index
    extrapolated_tb = (mean_mb * n_for_extrap) / 1e6  # MB → TB

    notes: list[str] = []
    if tier == "level2" and camera_fraction == 0:
        notes.append(
            "no camera groups detected in level-2 sample — consistent with "
            "the 2026-05-19 finding that cameras live in level-1"
        )
    if sustained_throughput_mbps < THROUGHPUT_DEGRADED_MBPS:
        notes.append(
            "sustained throughput below 50 MB/s — try more --numworkers or switch host"
        )

    report = ProbeReport(
        tier=tier,
        n_shots_in_index=n_index,
        n_shots_in_tier=n_for_extrap,
        sample_size=len(samples),
        samples=samples,
        sustained_throughput_mbps=sustained_throughput_mbps,
        median_shot_size_mb=median_mb,
        p95_shot_size_mb=p95_mb,
        extrapolated_total_size_tb=extrapolated_tb,
        group_coverage=coverage,
        camera_coverage_fraction=camera_fraction,
        started_at=started_at,
        finished_at=finished_at,
        notes=notes,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"probe-{tier}-{int(time.time())}.json"
        report_path.write_text(report.to_json(), encoding="utf-8")
        report.notes.append(f"report written to {report_path}")

    return report
