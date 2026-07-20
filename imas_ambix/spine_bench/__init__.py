"""Physics-spine benchmark — a durable, versioned perf + quality evolution metric.

This is the EQUILIBRIUM-ENGINE (reanalysis "spine") benchmark, distinct from the
world-model camera/rFID benchmark in :mod:`imas_ambix.bench`.  It stamps the current
engine's performance (per-slice solve wall, throughput) AND physics quality (grid-free
vs grid-GS reproduction, the flux-surface-averaging d-roughness that motivates
greens-filament-solver §3, convergence/confinement health) on a FROZEN named shot set,
persisting a schema-versioned YAML record keyed by git commit + machine.

Design goals (so it can be relied on as a real evolution metric):
  * FROZEN shot set + versioned metrics + engine-config SHA → runs stay comparable
    across time and commits.  Bump ``SHOTSET_VERSION`` / ``SCHEMA_VERSION`` when either
    changes, never silently.
  * asv-inspired (commit/machine keying, warmup+repeat timing stats, track-style
    arbitrary quality metrics) without asv's isolated-env-per-commit model, which does
    not fit a heavy-data / GPU physics solve.  The YAML records are asv-wrappable later
    (see README) and GHCR-portable (self-describing, commit+machine keyed).
"""

from imas_ambix.spine_bench.schema import (
    METRICS,
    SCHEMA_VERSION,
    EnvInfo,
    MachineInfo,
    Metric,
    ShotStamp,
    SpineBenchmarkStamp,
)
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET, SHOTSET_VERSION, BenchShot

__all__ = [
    "METRICS",
    "SCHEMA_VERSION",
    "SHOTSET_VERSION",
    "FROZEN_SHOTSET",
    "BenchShot",
    "Metric",
    "MetricValue",
    "MachineInfo",
    "EnvInfo",
    "ShotStamp",
    "SpineBenchmarkStamp",
]
