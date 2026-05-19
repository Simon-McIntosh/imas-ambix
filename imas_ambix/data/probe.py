"""FAIR-MAST sizing probe.

Implements the protocol in ``plans/data-acquisition.md`` §3:

1. Pull the parquet shot index.
2. Sample N shots, run ``s5cmd cp`` for each, measure size + throughput.
3. Report acceptance against the configured thresholds.

Run from a ``sirius`` standard compute node (outbound network and GPFS
access). The GPU node ``betelgeuse`` has no outbound network and will fail
silently before the first object is fetched.
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
    S3_BUCKET,
    S3_ENDPOINT,
)

# Acceptance thresholds — see plans/data-acquisition.md §3.2
THROUGHPUT_MIN_MBPS = 200.0
THROUGHPUT_DEGRADED_MBPS = 50.0
SHOT_P95_MAX_GB = 5.0
CAMERA_SHOTS_MIN = 1000
TOTAL_SIZE_MIN_TB = 2.0
TOTAL_SIZE_MAX_TB = 12.0


@dataclass
class ShotSample:
    """Per-shot result from the probe."""

    shot_id: int
    has_camera: bool
    bytes_copied: int
    elapsed_s: float
    error: str | None = None

    @property
    def throughput_mbps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return (self.bytes_copied / 1e6) / self.elapsed_s


@dataclass
class ProbeReport:
    """Aggregate probe result, serialisable to JSON."""

    n_shots_in_index: int
    n_camera_shots: int
    sample_size: int
    samples: list[ShotSample] = field(default_factory=list)
    sustained_throughput_mbps: float = 0.0
    median_shot_size_mb: float = 0.0
    p95_shot_size_mb: float = 0.0
    extrapolated_total_size_tb: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "n_shots_in_index": self.n_shots_in_index,
            "n_camera_shots": self.n_camera_shots,
            "sample_size": self.sample_size,
            "sustained_throughput_mbps": self.sustained_throughput_mbps,
            "median_shot_size_mb": self.median_shot_size_mb,
            "p95_shot_size_mb": self.p95_shot_size_mb,
            "extrapolated_total_size_tb": self.extrapolated_total_size_tb,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": self.notes,
            "acceptance": self.acceptance_summary(),
            "samples": [
                {
                    "shot_id": s.shot_id,
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

        out["camera_shots"] = (
            "pass" if self.n_camera_shots >= CAMERA_SHOTS_MIN else "fail"
        )
        out["per_shot_p95"] = (
            "pass" if self.p95_shot_size_mb <= SHOT_P95_MAX_GB * 1024 else "fail"
        )
        return out


def s5cmd_available() -> bool:
    """Return True if the ``s5cmd`` binary is on PATH."""
    return shutil.which("s5cmd") is not None


def run_shot_copy(
    shot_id: int,
    dest_root: Path,
    numworkers: int = 32,
    timeout_s: float = 600.0,
) -> ShotSample:
    """Copy a single shot via ``s5cmd`` and return a :class:`ShotSample`.

    Bytes are measured by ``du``-ing the destination after copy. Errors
    from ``s5cmd`` are captured on the sample, not raised.
    """
    dest = dest_root / f"{shot_id}.zarr"
    dest.mkdir(parents=True, exist_ok=True)
    src = f"s3://{S3_BUCKET}/level2/shots/{shot_id}.zarr/*"
    cmd = [
        "s5cmd",
        "--no-sign-request",
        "--endpoint-url",
        S3_ENDPOINT,
        "--numworkers",
        str(numworkers),
        "cp",
        src,
        str(dest) + "/",
    ]
    t0 = time.monotonic()
    error: str | None = None
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
    return ShotSample(
        shot_id=shot_id,
        has_camera=False,  # caller patches this after constructing samples
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
    camera_only: bool,
    seed: int = 0,
) -> list[int]:
    """Return a deterministic shot-id sample from the index."""
    from imas_ambix.data.manifest import filter_camera_bearing

    pool = filter_camera_bearing(df) if camera_only else df
    if len(pool) == 0:
        return []
    n = min(sample_size, len(pool))
    return [int(s) for s in pool.sample(n=n, random_state=seed)["shot_id"].tolist()]


def run_probe(
    sample_size: int = 50,
    numworkers: int = 32,
    timeout_s: float = 600.0,
    output_dir: Path | None = None,
    keep_samples: bool = False,
    camera_only: bool = False,
    seed: int = 0,
) -> ProbeReport:
    """Run the full probe end-to-end and return the populated report.

    Heavy lifting (pandas import, ``s5cmd`` invocation, ``du``) lives
    inside this function so that importing the module is cheap.
    """
    from imas_ambix.data.manifest import (
        detect_camera_columns,
        filter_camera_bearing,
        load_index,
    )

    if not s5cmd_available():
        raise RuntimeError(
            "s5cmd is not on PATH — see plans/data-acquisition.md §3.1 for the "
            "install command (`pip install --user s5cmd`)"
        )

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    df = load_index()
    camera_cols = detect_camera_columns(df)
    n_camera = len(filter_camera_bearing(df)) if camera_cols else 0

    sample_ids = sample_shots(df, sample_size, camera_only, seed=seed)
    samples: list[ShotSample] = []

    with tempfile.TemporaryDirectory(prefix="mast-probe-") as tmp:
        tmp_path = Path(tmp)
        for shot_id in sample_ids:
            s = run_shot_copy(
                shot_id, tmp_path, numworkers=numworkers, timeout_s=timeout_s
            )
            # Lookup camera flag for this shot.
            row = df[df["shot_id"] == shot_id]
            if camera_cols and not row.empty:
                s.has_camera = bool(
                    row[list(camera_cols)]
                    .fillna(False)
                    .astype(bool)
                    .any(axis=1)
                    .iloc[0]
                )
            samples.append(s)
            # Move into the persistent output directory rather than tmp.
            if keep_samples and output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                shot_path = tmp_path / f"{shot_id}.zarr"
                if shot_path.exists():
                    shutil.move(str(shot_path), str(output_dir / f"{shot_id}.zarr"))

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

    # Extrapolate total size to the whole index — separately if camera_only,
    # else from the full count.
    n_shots = len(df)
    extrapolated_tb = (mean_mb * n_shots) / 1e6  # MB → TB

    notes: list[str] = []
    if not camera_cols:
        notes.append(
            "no camera-flag columns detected on the index — camera_shots gate is "
            "set to 0 by definition; re-evaluate after the mirror lands."
        )
    if sustained_throughput_mbps < THROUGHPUT_DEGRADED_MBPS:
        notes.append(
            "sustained throughput below 50 MB/s — open a ticket with STFC before "
            "the bulk download."
        )

    report = ProbeReport(
        n_shots_in_index=n_shots,
        n_camera_shots=n_camera,
        sample_size=len(samples),
        samples=samples,
        sustained_throughput_mbps=sustained_throughput_mbps,
        median_shot_size_mb=median_mb,
        p95_shot_size_mb=p95_mb,
        extrapolated_total_size_tb=extrapolated_tb,
        started_at=started_at,
        finished_at=finished_at,
        notes=notes,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"probe-{int(time.time())}.json"
        report_path.write_text(report.to_json(), encoding="utf-8")

    return report
