"""Run the physics-spine benchmark on the frozen shot set → a versioned YAML stamp.

Reuses the VALIDATED greens-filament-solver §2 gate solve path (the basin
solve + profile solve, both substrates) so the benchmark measures exactly
the engine the gate proved, not a parallel re-derivation.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from pathlib import Path

import numpy as np

from imas_ambix.spine_bench.schema import (
    SCHEMA_VERSION,
    EnvInfo,
    MachineInfo,
    ShotStamp,
    SpineBenchmarkStamp,
)
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET, resolve_shotset_version

CONFINED_AXIS_R_MAX = 1.4
_NRHO_SWEEP = (16, 32, 64, 96)


# --- metadata ---------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _machine_info() -> MachineInfo:
    import os

    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:  # noqa: BLE001
        pass
    ram_gb = 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        ram_gb = round(pages * page / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    return MachineInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        cpu_model=cpu_model,
        n_logical_cpu=os.cpu_count() or 0,
        ram_gb=ram_gb,
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
    )


def _env_info(engine_cfg_name: str, engine_cfg_sha: str) -> EnvInfo:
    import scipy

    dirty = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))
    return EnvInfo(
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        git_commit=_run(["git", "rev-parse", "HEAD"]) or "unknown",
        git_dirty=dirty,
        engine_config_name=engine_cfg_name,
        engine_config_sha=engine_cfg_sha,
    )


# --- the FSA quality metric (the §3 motivation) -----------------------------


def _d_roughness(d_face: np.ndarray) -> float:
    """Relative second-difference roughness of the diffusion coefficient d = g2·g3/ρ̂.

    ``rms(Δ²d) / rms(d)`` over the interior faces (axis zero dropped) — a
    resolution-comparable, dimensionless measure of the noise the resistive diffusion
    integrates against. Higher = noisier; on the grid-GS substrate it rises with n_rho.
    """
    d = np.asarray(d_face, dtype=np.float64)
    d = d[1:]  # drop the on-axis regular-limit zero
    d = d[np.isfinite(d)]
    if d.size < 4:
        return float("nan")
    scale = float(np.sqrt(np.mean(d**2))) or 1.0
    d2 = np.diff(d, n=2)
    return float(np.sqrt(np.mean(d2**2)) / scale)


#: The FSA reads compared head-to-head: the host coarea binning (baseline) and
#: the accelerator-native, contour-free JAX kernel-coarea (greens-filament-solver §3).
_FSA_MODES = ("coarea", "connectivity")


def _fsa_roughness_sweep(
    fit, grid, bt0, *, n_p, n_f, nonneg
) -> dict[str, dict[int, float]]:
    """d-roughness at each n_rho per FSA mode, from a fit's force-balanced ψ.

    Returns ``{mode: {n_rho: roughness}}`` for both the host coarea read and the
    contour-free connectivity read on the SAME equilibrium — the only difference
    is how the flux surfaces are averaged, so the two are directly comparable.
    """
    from imas_ambix.latent.current_diffusion import flux_surface_geometry

    out: dict[str, dict[int, float]] = {m: {} for m in _FSA_MODES}
    if not (fit.scored and fit.psi is not None and fit.coeffs is not None):
        return out
    if not (np.isfinite(fit.target[0]) and float(fit.target[0]) <= CONFINED_AXIS_R_MAX):
        return out
    for mode in _FSA_MODES:
        for n_rho in _NRHO_SWEEP:
            geo = flux_surface_geometry(
                fit.psi,
                grid,
                coeffs=np.asarray(fit.coeffs, dtype=np.float64),
                ip_amperes=abs(float(fit.ip_amperes)),
                n_p=n_p,
                n_f=n_f,
                nonneg=nonneg,
                b_phi0=bt0,
                n_rho=n_rho,
                fsa_mode=mode,
            )
            if geo is None:
                continue
            r = _d_roughness(np.asarray(geo.d_face))
            if np.isfinite(r):
                out[mode][n_rho] = r
    return out


# --- the sensor-space misfit (reconstruction vs measurement) ----------------


def _magnetics_residual(fit, grid, table, payload) -> float:
    """Whitened field/flux misfit of one converged equilibrium against its slice.

    ``rms((pred - meas)/scale)`` over the channels the payload marks mapped and
    measured, with ``pred`` the known-coil vacuum prediction plus the
    cell→sensor Green's matvec of the converged plasma cell currents.  Rows of
    that Green's matrix follow ``table.sensor_map``, which is the order the
    payload's ``measured`` / ``vacuum`` / ``scale`` / ``mask`` vectors are built
    in, so no re-alignment is needed.

    The payload's own mask is used, not the all-off mask the frozen spine solves
    under: these channels are never consumed by the solve, which is what makes
    the number an independent forward-model check.  Sensor positions, probe
    orientations, coil filaments and the limiter all enter through ``grid`` and
    ``table``, so a change of geometry source shows up here.
    """
    if not (fit.scored and fit.jphi_flat is not None):
        return float("nan")
    mask = np.asarray(payload.mask, dtype=bool)
    if not mask.any():
        return float("nan")
    g_sens, _channels = grid.sensor_greens(table)
    i_cell = np.asarray(fit.jphi_flat, dtype=np.float64)[grid.cells] * (
        grid.dr * grid.dz
    )
    pred = np.asarray(payload.vacuum, dtype=np.float64) + g_sens @ i_cell
    meas = np.asarray(payload.measured, dtype=np.float64)
    scale = np.asarray(payload.scale, dtype=np.float64)
    ok = mask & np.isfinite(meas) & np.isfinite(pred) & (scale > 0.0)
    if not ok.any():
        return float("nan")
    resid = (pred[ok] - meas[ok]) / scale[ok]
    return float(np.sqrt(np.mean(resid * resid)))


def _median(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


# --- the benchmark ----------------------------------------------------------


def run_stamp(
    *,
    created_utc: str,
    max_slices: int = 6,
    sigma: float = 0.02,
    shots=None,
    topology_reads: tuple[str, ...] = ("hard",),
) -> SpineBenchmarkStamp:
    """Solve the frozen shot set under both substrates and assemble the stamp.

    ``topology_reads`` adds per-sweep topology-read arms on the PRIMARY
    (greens-matvec) substrate: ``("hard",)`` is the historical two-arm stamp;
    include ``"connectivity"`` to bench the continuous (smooth-map) read
    head-to-head, with its reproduction metrics vs the hard-read solve.

    ``shots`` overrides the frozen set. The stamp is then labelled by
    :func:`resolve_shotset_version`, so an override cannot be written under the
    frozen metric's name.
    """
    from imas_ambix.latent.boundary_disc import disc_read
    from imas_ambix.latent.gs_solve import (
        SUBSTRATE_GREENS,
        SUBSTRATE_GRID,
        EquilibriumGrid,
    )
    from scripts.heldout_mse_gate_eval import _campaign_table, shot_bt0
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import factory_shot_payloads, frozen_spine_config

    shotset = shots if shots is not None else FROZEN_SHOTSET
    shotset_version = resolve_shotset_version(shots)
    spine, spine_sha = frozen_spine_config()
    iso = spine["interior_solve"]
    n_p, n_f = int(iso["n_p"]), int(iso["n_f"])
    nonneg = iso["profile_kind"] == "monomial-nonneg"
    smoothness = float(iso["smoothness"])
    boundary_read = iso["boundary_read_scoring"]
    spine_kw = dict(
        n_p=n_p,
        n_f=n_f,
        nonneg=nonneg,
        smoothness=smoothness,
        boundary_read=boundary_read,
        sigma=sigma,
    )

    import resource
    import time

    from scripts.greens_filament_gate_eval import (
        _axis_cm,
        _fit_slice,
        _lcfs_cm,
        _profile,
        _profile_rms,
    )

    def _fit(g, sub, p, centroid, warm, **kw):
        return _fit_slice(
            g, tbl, basis, p, substrate=sub, warm=warm, centroid=centroid, **kw
        )

    run_t0 = time.perf_counter()
    stamps: list[ShotStamp] = []
    for bs in shotset:
        shot = int(bs.shot_id)
        table = _campaign_table(shot)
        if table is None:
            continue
        payload = factory_shot_payloads(
            shot, nr=65, nz=97, max_slices=max_slices, min_ip_ka=200.0, table=table
        )
        if payload is None:
            continue
        grid_gs = payload["grid"]
        tbl, basis = payload["table"], payload["basis"]
        grid_free = EquilibriumGrid.from_table(tbl, nr=65, nz=97)
        bt0 = shot_bt0(shot)
        sig = tbl.signature.key if hasattr(tbl, "signature") else ""
        order = np.argsort([p.time_s for p in payload["payloads"]])

        # per-substrate accumulators (phase timing + e2e latency + quality)
        def _acc():
            return {
                "fits": [],
                "conf": [],
                "conv": [],
                "rough": [],
                "mag_resid": [],
                "e2e": [],
                "ph_disc": [],
                "ph_basin": [],
                "ph_profile": [],
                "ph_fsa": [],
            }

        # benchmark arms: (key, substrate, topology_read, grid).  The topology-
        # read sweep applies to the PRIMARY (greens-matvec) substrate only.
        arms = [
            (SUBSTRATE_GRID, SUBSTRATE_GRID, "hard", grid_gs),
            (SUBSTRATE_GREENS, SUBSTRATE_GREENS, "hard", grid_free),
        ]
        for tr in topology_reads:
            if tr != "hard":
                arms.append(
                    (f"{SUBSTRATE_GREENS}+{tr}", SUBSTRATE_GREENS, tr, grid_free)
                )
        acc = {key: _acc() for key, _s, _t, _g in arms}
        for pos, k in enumerate(order):
            p = payload["payloads"][int(k)]
            t = time.perf_counter()
            inv = disc_read(p, grid_gs, tbl, basis)  # substrate-independent seed
            t_disc = time.perf_counter() - t
            if inv is None or inv.ring is None:
                continue
            centroid = (float(inv.centroid_r), float(inv.centroid_z))
            disc_seed = _disc_seed_flat(grid_gs, inv)
            for key, sub, tr, g in arms:
                # phase 1: basin solve (free-sign amplitude pair, centroid-pinned)
                t = time.perf_counter()
                f_basin = _fit(
                    g,
                    sub,
                    p,
                    centroid,
                    disc_seed,
                    n_p=1,
                    n_f=1,
                    nonneg=False,
                    smoothness=smoothness,
                    boundary_read=boundary_read,
                    sigma=sigma,
                    topology_read=tr,
                )
                t_basin = time.perf_counter() - t
                basin_ok = (
                    f_basin.scored
                    and f_basin.jphi_flat is not None
                    and np.isfinite(f_basin.target[0])
                    and f_basin.target[0] <= CONFINED_AXIS_R_MAX
                )
                seed = f_basin.jphi_flat if basin_ok else disc_seed
                # phase 2: profile solve (non-negative ladder — the readout equilibrium)
                t = time.perf_counter()
                fit = _fit(g, sub, p, centroid, seed, topology_read=tr, **spine_kw)
                t_profile = time.perf_counter() - t
                # phase 3: FSA readout (timed as a component)
                t = time.perf_counter()
                rough = (
                    _fsa_roughness_sweep(fit, g, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
                    if fit.scored
                    else {}
                )
                t_fsa = time.perf_counter() - t

                a = acc[key]
                a["fits"].append((int(k), fit))
                if pos > 0:  # discard each shot's first slice as timing warmup
                    a["ph_disc"].append(t_disc * 1e3)
                    a["ph_basin"].append(t_basin * 1e3)
                    a["ph_profile"].append(t_profile * 1e3)
                    a["ph_fsa"].append(t_fsa * 1e3)
                    a["e2e"].append((t_disc + t_basin + t_profile) * 1e3)
                if fit.scored:
                    a["conv"].append(1.0)
                    cr = float(fit.target[0]) if np.isfinite(fit.target[0]) else 1e9
                    a["conf"].append(1.0 if cr <= CONFINED_AXIS_R_MAX else 0.0)
                    if rough:
                        a["rough"].append(rough)
                    mag = _magnetics_residual(fit, g, tbl, p)
                    if np.isfinite(mag):
                        a["mag_resid"].append(mag)
                else:
                    a["conv"].append(0.0)

        # reproduction: the greens-matvec DEV SPINE vs the grid-Δ* baseline check
        gs_by_k = dict(acc[SUBSTRATE_GRID]["fits"])
        fr_by_k = dict(acc[SUBSTRATE_GREENS]["fits"])
        axis_cm, lcfs_cm, prof_rms = [], [], []
        for k in gs_by_k:
            fg, ff = gs_by_k[k], fr_by_k.get(k)
            if ff is None or not (fg.scored and ff.scored):
                continue
            axis_cm.append(_axis_cm(fg.target, ff.target))
            lcfs_cm.append(_lcfs_cm(fg.target, ff.target))
            jg = _profile(fg, grid_gs, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
            jf = _profile(ff, grid_free, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
            prof_rms.append(_profile_rms(jg, jf))

        # reproduction: the continuous topology read vs the hard read, same
        # (greens-matvec) substrate — the smooth read must not be lossy
        smooth_rep: dict[str, list[float]] = {}
        for key, _sub, tr, _g in arms:
            if tr == "hard":
                continue
            sm_by_k = dict(acc[key]["fits"])
            sm_axis, sm_lcfs, sm_prof = [], [], []
            for k in fr_by_k:
                fh, fs = fr_by_k[k], sm_by_k.get(k)
                if fs is None or not (fh.scored and fs.scored):
                    continue
                sm_axis.append(_axis_cm(fh.target, fs.target))
                sm_lcfs.append(_lcfs_cm(fh.target, fs.target))
                jh = _profile(fh, grid_free, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
                js = _profile(fs, grid_free, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
                sm_prof.append(_profile_rms(jh, js))
            smooth_rep[key] = [
                _median(sm_axis),
                _median(sm_lcfs),
                _median(sm_prof),
            ]

        for key, sub, tr, _g in arms:
            a = acc[key]
            metrics: dict[str, float] = {}
            if a["ph_profile"]:
                profile_med = float(np.median(a["ph_profile"]))
                metrics["solve_wall_ms_per_slice"] = profile_med
                metrics["throughput_slices_per_core_s"] = 1000.0 / profile_med
            if a["e2e"]:
                metrics["end_to_end_ms_per_slice"] = float(np.median(a["e2e"]))
                metrics["latency_ms_p50"] = float(np.percentile(a["e2e"], 50))
                metrics["latency_ms_p99"] = float(np.percentile(a["e2e"], 99))
            if a["conv"]:
                metrics["converged_fraction"] = float(np.mean(a["conv"]))
            if a["conf"]:
                metrics["confined_fraction"] = float(np.mean(a["conf"]))
            if a["mag_resid"]:
                metrics["magnetics_residual_whitened_rms"] = _median(a["mag_resid"])
            _fsa_names = {
                "coarea": (
                    "fsa_d_roughness_nrho32",
                    "fsa_d_roughness_nrho96",
                    "fsa_d_roughness_resolution_slope",
                ),
                "connectivity": (
                    "fsa_d_roughness_conn_nrho32",
                    "fsa_d_roughness_conn_nrho96",
                    "fsa_d_roughness_conn_resolution_slope",
                ),
            }
            for mode, (n32, n96, nslope) in _fsa_names.items():
                per = [r[mode] for r in a["rough"] if mode in r]
                r32 = _median([r.get(32, np.nan) for r in per])
                r96 = _median([r.get(96, np.nan) for r in per])
                if np.isfinite(r32):
                    metrics[n32] = r32
                if np.isfinite(r96):
                    metrics[n96] = r96
                xs, ys = [], []
                for nr in _NRHO_SWEEP:
                    med = _median([r.get(nr, np.nan) for r in per])
                    if np.isfinite(med):
                        xs.append(np.log2(nr))
                        ys.append(med)
                if len(xs) >= 2:
                    metrics[nslope] = float(np.polyfit(xs, ys, 1)[0])
            if key == SUBSTRATE_GREENS:  # dev-spine vs the grid baseline check
                if axis_cm:
                    metrics["axis_reproduce_cm"] = _median(axis_cm)
                if lcfs_cm:
                    metrics["lcfs_reproduce_cm"] = _median(lcfs_cm)
                if prof_rms:
                    metrics["profile_reproduce_rms"] = _median(prof_rms)
            if key in smooth_rep:  # smooth read vs the hard read, same substrate
                sm_a, sm_l, sm_p = smooth_rep[key]
                if np.isfinite(sm_a):
                    metrics["axis_smoothread_cm"] = sm_a
                if np.isfinite(sm_l):
                    metrics["lcfs_smoothread_cm"] = sm_l
                if np.isfinite(sm_p):
                    metrics["profile_smoothread_rms"] = sm_p
            phase = {
                "disc_read": _median(a["ph_disc"]),
                "basin_solve": _median(a["ph_basin"]),
                "profile_solve": _median(a["ph_profile"]),
                "fsa_readout": _median(a["ph_fsa"]),
            }
            stamps.append(
                ShotStamp(
                    shot_id=shot,
                    role=bs.role,
                    campaign_signature=sig,
                    substrate=sub,
                    topology_read=tr,
                    n_slices_attempted=len(a["fits"]),
                    n_slices_scored=int(sum(a["conv"])),
                    timing_n_repeat=len(a["e2e"]),
                    metrics=metrics,
                    phase_timing_ms={k: v for k, v in phase.items() if np.isfinite(v)},
                )
            )

    # aggregate rollups: median of each metric across shots, per arm
    # (arm key = substrate, suffixed '+<topology_read>' for non-hard reads)
    def _arm_key(s: ShotStamp) -> str:
        return (
            s.substrate
            if s.topology_read == "hard"
            else f"{s.substrate}+{s.topology_read}"
        )

    agg: dict[str, dict[str, float]] = {}
    for arm in {_arm_key(s) for s in stamps}:
        keys = {k for s in stamps if _arm_key(s) == arm for k in s.metrics}
        agg[arm] = {
            k: _median(
                [s.metrics[k] for s in stamps if _arm_key(s) == arm and k in s.metrics]
            )
            for k in sorted(keys)
        }

    peak_rss_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    env = _env_info("closure-spine-D2", spine_sha)
    return SpineBenchmarkStamp(
        schema_version=SCHEMA_VERSION,
        shotset_version=shotset_version,
        created_utc=created_utc,
        complete_run_wall_s=round(time.perf_counter() - run_t0, 2),
        peak_rss_gb=peak_rss_gb,
        machine=_machine_info(),
        env=env,
        shots=stamps,
        aggregate=agg,
        notes=[
            "greens-matvec = the FILAMENT / single-interaction-matrix DEV SPINE "
            "(primary, GPU target); grid-delstar = the gridded delta* solve, "
            "retained as a baseline CHECK.",
            "Perf at OMP=1; per-shot first slice discarded as warmup. solve_wall = "
            "the profile solve only; end_to_end/latency = disc-seed + basin solve "
            "+ profile solve.",
            "Reproduction (axis/lcfs/profile) validates the greens-matvec dev spine "
            "against the grid-delta* baseline check on the same slice (greens row).",
            "magnetics_residual_whitened_rms scores the converged equilibrium "
            "against the RAW magnetics, which the frozen spine never fits (mask "
            "OFF), so it is a forward-model check on the machine geometry rather "
            "than a fit residual; its floor is the static coil / sensor "
            "calibration misfit.",
            "bench_scope = per-slice STATIC solve (the GPU inner-loop target); the "
            "dynamics-coupled label rollout (diffusion + passive + temporal "
            "warm-start) is a distinct mode to add before the corpus GPU run.",
            "FSA d-roughness = rms(2nd-diff of d_face)/rms(d_face), interior faces. "
            "RECONCILE: the greens-filament-solver s3 plan cites a cell-binned "
            "d-roughness ~0.5 worsening 0.45->0.72 with n_rho; on the current "
            "CONTOUR-INTEGRATED geo.d_face the slope measured here is <=0 — no "
            "committed diagnostic backs the plan's number, so s3 should re-baseline.",
            "Per-component / GPU-device memory is added with the GPU rollout; "
            "peak_rss_gb here is process-level. Held-out-MSE pitch is tracked "
            "separately (heldout_mse_gate_eval).",
            "CAVEAT (corpus extrapolation): per-slice metrics are on the CACHED "
            "Green's matrix (built once per shot, warmup-excluded), so a corpus-cost "
            "extrapolation from throughput_slices_per_core_s ASSUMES campaign-scope "
            "Green's caching (greens-filament-solver §4). Today the grid+matrices are "
            "rebuilt per shot (only ~2 campaign signatures exist), adding ~2% at "
            "corpus scale until §4 lands.",
        ],
    )


def write_yaml(stamp: SpineBenchmarkStamp, out_dir: Path) -> Path:
    """Persist the stamp as YAML keyed by commit + machine."""
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    commit = stamp.env.git_commit[:10]
    host = stamp.machine.hostname.split(".")[0]
    dirty = "-dirty" if stamp.env.git_dirty else ""
    path = (
        out_dir / f"physics-spine-{stamp.shotset_version}-{commit}{dirty}-{host}.yaml"
    )
    path.write_text(yaml.safe_dump(stamp.model_dump(), sort_keys=False, width=100))
    return path
