"""Versioned schema + metric registry for the physics-spine benchmark.

The schema is the contract that makes this a *comparable-over-time* evolution metric.
Every record carries the schema version, the shot-set version, the engine-config SHA,
and the machine + environment it was measured on, so two stamps are only compared when
those agree (or the difference is explicit).  Metrics are a NAMED REGISTRY with units
and a direction — a metric's meaning never drifts without a version bump.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Bump when the record shape or a metric definition changes (never silently).
#: 1.1 added the ``device`` dimension, phase timing, peak-RSS memory, end-to-end
#: latency percentiles, and the complete-run wall — the GPU-vs-CPU throughput/latency
#: story lives in these.
SCHEMA_VERSION = "spine-bench/1.1"


class Metric(BaseModel):
    """One named, unit-carrying, direction-carrying metric definition."""

    name: str
    unit: str
    direction: str = Field(description="'lower_better' or 'higher_better'")
    description: str


#: The metric registry — the single source of truth for what each number MEANS.
#: A stamp records values keyed by these names; a reader interprets them by this table.
METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        # --- performance (the GPU-rollout evolution metric) ---
        Metric(
            name="solve_wall_ms_per_slice",
            unit="ms/slice",
            direction="lower_better",
            description="Median wall-clock of one rich frozen-spine ladder solve "
            "(K=2 scaffold + rich non-negative ladder), warmup-excluded.",
        ),
        Metric(
            name="throughput_slices_per_core_s",
            unit="slice/(core·s)",
            direction="higher_better",
            description="Slices solved per single-core second (1/solve_wall at "
            "OMP=1) — the corpus-label-factory throughput proxy. Report the GPU "
            "aggregate (slices/s over all cards) against this CPU per-core figure.",
        ),
        Metric(
            name="end_to_end_ms_per_slice",
            unit="ms/slice",
            direction="lower_better",
            description="Median COMPLETE per-slice equilibrium wall: disc-read seed "
            "+ K=2 scaffold + rich ladder (not just the rich solve). The full cost "
            "of producing one labelled equilibrium.",
        ),
        Metric(
            name="latency_ms_p50",
            unit="ms",
            direction="lower_better",
            description="Median end-to-end per-slice wall — the streaming-latency "
            "proxy (true in-pulse stream latency = window-fill + solve + emit, "
            "measured by the gpu-accelerated-labeler in-pulse rung).",
        ),
        Metric(
            name="latency_ms_p99",
            unit="ms",
            direction="lower_better",
            description="p99 of the end-to-end per-slice wall — the tail latency "
            "a real-time control-room budget must clear.",
        ),
        Metric(
            name="peak_rss_gb",
            unit="GB",
            direction="lower_better",
            description="Peak process resident set size over the whole benchmark "
            "run (ru_maxrss). Process-level; per-component / GPU-device memory is "
            "added with the GPU rollout (device memory tools).",
        ),
        # --- physics quality: substrate reproduction (grid-free vs grid-GS) ---
        Metric(
            name="axis_reproduce_cm",
            unit="cm",
            direction="lower_better",
            description="Median |Δ| between the grid-free (greens-matvec) and "
            "grid-GS (delstar) magnetic-axis positions on the same slice.",
        ),
        Metric(
            name="lcfs_reproduce_cm",
            unit="cm",
            direction="lower_better",
            description="Median |Δ| over the 8 LCFS radii between grid-free and "
            "grid-GS on the same slice.",
        ),
        Metric(
            name="profile_reproduce_rms",
            unit="normalised",
            direction="lower_better",
            description="Median normalised RMS between the grid-free and grid-GS "
            "jphi(rho_hat) profile on the same slice (well-determined slices).",
        ),
        # --- physics quality: FSA integrity (greens-filament-solver §3 motivation) ---
        Metric(
            name="fsa_d_roughness_nrho32",
            unit="normalised",
            direction="lower_better",
            description="Relative second-difference roughness of the diffusion "
            "coefficient d = g2·g3/rho_hat (geo.d_face) at n_rho=32 — the noisy "
            "flux-surface metric that corrupts the resistive diffusion.",
        ),
        Metric(
            name="fsa_d_roughness_nrho96",
            unit="normalised",
            direction="lower_better",
            description="Same d-roughness at n_rho=96; compare to nrho32 for the "
            "resolution trend (the greens-filament-solver §3 plan claims the "
            "cell-binned FSA worsens 0.45→0.72 with n_rho — this metric measures "
            "the trend on the actual contour-integrated geo.d_face).",
        ),
        Metric(
            name="fsa_d_roughness_resolution_slope",
            unit="Δrough/Δlog2(n_rho)",
            direction="lower_better",
            description="Slope of d-roughness vs log2(n_rho) over {16,32,64,96}; "
            "positive = degrades with resolution (the §3-claimed pathology), "
            "≤0 = stable/improving. Measured, not assumed.",
        ),
        # --- solve health ---
        Metric(
            name="converged_fraction",
            unit="fraction",
            direction="higher_better",
            description="Fraction of attempted slices whose solve converged below "
            "the frozen-spine convergence limit.",
        ),
        Metric(
            name="confined_fraction",
            unit="fraction",
            direction="higher_better",
            description="Fraction of scored slices with an inboard confined axis "
            "(R_axis ≤ the confined-axis cap).",
        ),
    ]
}


class MachineInfo(BaseModel):
    """Where the stamp was measured — comparability guard."""

    hostname: str
    platform: str
    cpu_model: str = ""
    n_logical_cpu: int = 0
    ram_gb: float = 0.0
    slurm_partition: str | None = None
    omp_num_threads: str | None = None


class EnvInfo(BaseModel):
    """Code + config identity of the measured engine."""

    python_version: str
    numpy_version: str = ""
    scipy_version: str = ""
    git_commit: str
    git_dirty: bool
    engine_config_name: str = ""
    engine_config_sha: str = ""
    device: str = "cpu"  # 'cpu' | 'gpu' — the comparison axis GPU stamps set to 'gpu'
    backend: str = "numpy-scipy"  # 'numpy-scipy' | 'jax-cpu' | 'jax-gpu'
    #: The solvers benched. PRIMARY = the filament / single-interaction-matrix solve
    #: (greens-matvec: analytic ψ = G·I, the pivot away from the gridded Δ*), our most
    #: technically-able solver and the GPU target. BASELINE CHECK = the gridded Δ* solve
    #: (grid-delstar), retained as the trusted numerical reference the filament solve is
    #: validated against — not the production target. Both at closure-spine-D2
    #: (disc seed + R/Z centroid, rich non-negative K=3+3, smoothness ridge, pushout).
    solver: str = (
        "free-boundary GS ladder (solve_equilibrium_lsq @ closure-spine-D2); "
        "PRIMARY = greens-matvec filament/single-matrix, "
        "BASELINE-CHECK = grid-delstar delta*"
    )


class ShotStamp(BaseModel):
    """One frozen benchmark shot's result under one substrate."""

    shot_id: int
    role: str
    campaign_signature: str = ""
    substrate: str  # 'grid-delstar' | 'greens-matvec'
    n_slices_attempted: int = 0
    n_slices_scored: int = 0
    timing_n_repeat: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    #: Median per-phase wall [ms] — disc_read / scaffold_k2 / rich_ladder / fsa_readout;
    #: the component breakdown that attributes where the solve time goes.
    phase_timing_ms: dict[str, float] = Field(default_factory=dict)


class SpineBenchmarkStamp(BaseModel):
    """The full versioned benchmark record — one YAML per (commit, machine, device)."""

    schema_version: str = SCHEMA_VERSION
    shotset_version: str
    benchmark_name: str = "physics-spine"
    created_utc: str  # ISO-8601, stamped by the caller
    #: What is timed. The per-slice static solve is the GPU inner-loop target; the
    #: dynamics-coupled label rollout (§3 diffusion + passive + temporal warm-start)
    #: is the label-factory throughput — a distinct mode, added before the corpus run.
    bench_scope: str = (
        "per-slice static free-boundary solve (K=2 scaffold + rich ladder)"
    )
    complete_run_wall_s: float = 0.0  # end-to-end wall of the whole benchmark run
    peak_rss_gb: float = 0.0  # peak process RSS over the run (ru_maxrss); run-level
    machine: MachineInfo
    env: EnvInfo
    shots: list[ShotStamp] = Field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
