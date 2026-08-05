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
#: story lives in these.  1.2 added the CONNECTIVITY (accelerator-native, contour-free
#: JAX) flux-surface-averaging d-roughness alongside the host coarea baseline, so a
#: single stamp compares the two FSA reads head-to-head.
#: 1.3 added the ``topology_read`` dimension (hard host read vs the continuous
#: connectivity/smooth-mask read that makes the fixed-point map differentiable),
#: its reproduction metrics, and the batched-GPU foundation metrics (GEMM
#: crossover + batched on-device inner-solve throughput/reproduction/precision).
#: 1.4 added ``magnetics_residual_whitened_rms`` — the sensor-space field/flux
#: misfit of the converged equilibrium, the first metric that scores the
#: reconstruction against the measurement rather than against another solve.
#: 1.5 added ``geometry_source`` + ``geometry_provenance``: which description of
#: the machine supplied the geometry, and its identity.  The sensor-space misfit
#: moves when the machine behind the Green's functions moves, so two stamps that
#: differ only in geometry source are measuring different things — and before
#: this field the only trace of that was the signature string, which a reader had
#: to recognise.  A run now states its source rather than being identified by it.
#: 1.6 added ``measurement_read``: which acquisition range setting was divided out
#: of each magnetics channel before the misfit was formed.  Nineteen probe channels
#: were recorded at more than one setting, so this had to be treated as a possible
#: break in what the residual MEANS — and both arms were re-stamped together at one
#: commit to find out.  They came back bit-identical to their 1.5 values: on all
#: six frozen shots every channel with a measured setting sits in a block recorded
#: at the reference rung, so the read divides by exactly one.  The frozen residual
#: is therefore CONTINUOUS across this bump and 1.5 stamps stay directly
#: comparable.  What changed is that the read now guarantees the amplitude
#: convention this curated shot set happened to satisfy, and each stamp says so.
SCHEMA_VERSION = "spine-bench/1.6"


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
            description="Median wall-clock of one frozen-spine profile solve "
            "(basin solve + profile solve), warmup-excluded.",
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
            "+ basin solve + profile solve (not just the profile solve). The full cost "
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
        # --- physics quality: FSA integrity ---
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
            "resolution trend. A cell-binned FSA is expected to worsen with n_rho; "
            "this metric measures the trend on the contour-integrated geo.d_face, "
            "where it does not.",
        ),
        Metric(
            name="fsa_d_roughness_resolution_slope",
            unit="Δrough/Δlog2(n_rho)",
            direction="lower_better",
            description="Slope of d-roughness vs log2(n_rho) over {16,32,64,96}; "
            "positive = degrades with resolution, "
            "≤0 = stable/improving. Measured, not assumed.",
        ),
        # --- FSA integrity: CONNECTIVITY (accelerator-native, contour-free) read ---
        Metric(
            name="fsa_d_roughness_conn_nrho32",
            unit="normalised",
            direction="lower_better",
            description="d = g2·g3/rho_hat roughness at n_rho=32 for the "
            "CONNECTIVITY FSA — the contour-free, fixed-shape JAX kernel-coarea "
            "read (flood-fill core + Gaussian surface averages, jit/vmap-safe). "
            "Compare head-to-head with fsa_d_roughness_nrho32 (host coarea).",
        ),
        Metric(
            name="fsa_d_roughness_conn_nrho96",
            unit="normalised",
            direction="lower_better",
            description="Connectivity-FSA d-roughness at n_rho=96; compare to "
            "conn_nrho32 for the resolution trend and to the coarea baseline.",
        ),
        Metric(
            name="fsa_d_roughness_conn_resolution_slope",
            unit="Δrough/Δlog2(n_rho)",
            direction="lower_better",
            description="Slope of the connectivity-FSA d-roughness vs log2(n_rho) "
            "over {16,32,64,96}; ≤0 = resolution-stable.",
        ),
        # --- topology-read reproduction (continuous/smooth vs hard host read) ---
        Metric(
            name="axis_smoothread_cm",
            unit="cm",
            direction="lower_better",
            description="Median |Δ| between the continuous-topology-read "
            "(connectivity binding + stencil axis + smooth core weight) and the "
            "hard-read magnetic-axis positions on the same slice, greens-matvec "
            "substrate.",
        ),
        Metric(
            name="lcfs_smoothread_cm",
            unit="cm",
            direction="lower_better",
            description="Median |Δ| over the 8 LCFS radii between the "
            "continuous-topology-read and hard-read solves on the same slice.",
        ),
        Metric(
            name="profile_smoothread_rms",
            unit="normalised",
            direction="lower_better",
            description="Median normalised RMS between the continuous-topology-"
            "read and hard-read jphi(rho_hat) profiles on the same slice.",
        ),
        # --- batched-GPU foundation (GEMM crossover + on-device inner solve) ---
        Metric(
            name="gpu_gemm_slices_per_s_b512",
            unit="slice/s",
            direction="higher_better",
            description="Batched Green's GEMM (psi = G·I) slices/s at B=512 on "
            "one GPU, fp64 unless the stamp notes a dtype.",
        ),
        Metric(
            name="gpu_gemm_crossover_batch",
            unit="batch",
            direction="lower_better",
            description="Batch size where the Green's matvec→GEMM becomes "
            "compute-bound (slices/s stops scaling ~linearly with B).",
        ),
        Metric(
            name="gpu_batched_solve_slices_per_s_b512",
            unit="slice/s",
            direction="higher_better",
            description="Batched fixed-shape on-device inner GS solve (vmap, "
            "fixed sweep budget, on-device topology read) slices/s at B=512 on "
            "one GPU.",
        ),
        Metric(
            name="gpu_batched_axis_reproduce_cm",
            unit="cm",
            direction="lower_better",
            description="Median |Δ| between the batched on-device inner-solve "
            "axis and the per-slice CPU grid-free reference axis (unique slices).",
        ),
        Metric(
            name="gpu_bf16_axis_delta_cm",
            unit="cm",
            direction="lower_better",
            description="Median axis shift when the inner-solve GEMM drops to "
            "bf16 (fp32 accumulate) vs the fp64 on-device baseline; topology "
            "read and cross-iteration state stay >= fp32/fp64.",
        ),
        # --- physics quality: sensor-space misfit against the measurement ---
        Metric(
            name="magnetics_residual_whitened_rms",
            unit="normalised",
            direction="lower_better",
            description="Median over scored slices of the whitened field/flux "
            "residual rms((pred - meas)/scale) over the slice's mapped and "
            "measured magnetics channels, where pred = the known-coil vacuum "
            "prediction + the cell-to-sensor Green's matvec of the converged "
            "plasma cell currents (jphi over in-limiter cells x cell area), "
            "meas = the amb magnetics with each channel's acquisition range "
            "setting divided out (schema 1.6; earlier stamps scored the recorded "
            "amplitudes, which step by up to a factor of two across the frozen "
            "set on nineteen channels), and scale = the per-channel robust "
            "signal scale that puts flux loops [Wb] and B-probes [T] into one "
            "dimensionless population. Every other physics metric compares one "
            "solve against another solve; this one compares the reconstruction "
            "against the measurement, so it is the metric that moves when the "
            "machine geometry behind the Green's functions moves. The frozen "
            "spine solves with the magnetics mask OFF (it consumes Ip, the "
            "measured current centroid and the source-free boundary read), so "
            "these channels are never fitted and the number is a forward-model "
            "check rather than a fit residual: its floor is the static coil and "
            "sensor-calibration misfit, not the solver tolerance.",
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
    #: (disc seed + R/Z centroid, non-negative profile ladder n_p=n_f=3,
    #: smoothness ridge, pushout).
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
    #: per-sweep topology read: 'hard' (host critical points + labelled mask) or
    #: 'connectivity' (continuous binding + stencil axis + smooth core weight)
    topology_read: str = "hard"
    n_slices_attempted: int = 0
    n_slices_scored: int = 0
    timing_n_repeat: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    #: Median per-phase wall [ms] — disc_read / basin_solve / profile_solve /
    #: fsa_readout;
    #: the component breakdown that attributes where the solve time goes.
    phase_timing_ms: dict[str, float] = Field(default_factory=dict)


class SpineBenchmarkStamp(BaseModel):
    """The full versioned benchmark record — one YAML per (commit, machine, device)."""

    schema_version: str = SCHEMA_VERSION
    shotset_version: str
    benchmark_name: str = "physics-spine"
    created_utc: str  # ISO-8601, stamped by the caller
    #: What is timed. The per-slice static solve is the GPU inner-loop target; the
    #: dynamics-coupled label rollout (diffusion + passive + temporal warm-start)
    #: is the label-factory throughput — a distinct mode, added before the corpus run.
    bench_scope: str = (
        "per-slice static free-boundary solve (basin solve + profile solve)"
    )
    complete_run_wall_s: float = 0.0  # end-to-end wall of the whole benchmark run
    peak_rss_gb: float = 0.0  # peak process RSS over the run (ru_maxrss); run-level
    #: Which description of the machine supplied the geometry behind the Green's
    #: functions.  ``'efm-campaign'`` is the campaign's own static arrays (the
    #: historical default, which is why it is the default here — stamps written
    #: before this field carry it implicitly); a machine-description artifact
    #: names itself instead.  Two stamps are only a like-for-like comparison of
    #: the ENGINE when this agrees; when it differs they measure a change of
    #: machine description, which is the wider-budget comparison in
    #: :mod:`imas_ambix.spine_bench.parity`.
    geometry_source: str = "efm-campaign"
    #: Which REVISION of that source, when the source is revised independently of
    #: the engine.  This is the ONLY field that separates two republications of one
    #: machine description: the setup signature hashes conductor positions, sensor
    #: positions and the limiter, so a republication that restates a TURN COUNT
    #: moves the forward model — and therefore the sensor-space misfit — while
    #: leaving the signature, the physical digest and the shot set all unchanged.
    #: A stamp identified only by its signature is ambiguous across such revisions.
    #: Empty when the source has no revision of its own.
    geometry_revision: str = ""
    #: The identity of that description — for an artifact, the digest triple and
    #: evidence state resolution verified before any IDS was opened.  Free-form
    #: because each source has its own identity; recorded so a stamp can be
    #: traced back to the exact machine it measured rather than to a cache path
    #: that may since have been repopulated.
    geometry_provenance: dict[str, object] = Field(default_factory=dict)
    #: What the measurement read did to the magnetics channels before the misfit
    #: was formed: which acquisition range settings were divided out, on which
    #: shots, and what warranted each one.  The geometry fields above say which
    #: machine was predicted; this says which measurement it was scored against,
    #: and the two together are what make a residual reproducible.  Empty for a
    #: run that read the archive exactly as published — which every stamp before
    #: schema 1.6 did, and none of them could say so.
    measurement_read: dict[str, object] = Field(default_factory=dict)
    machine: MachineInfo
    env: EnvInfo
    shots: list[ShotStamp] = Field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
