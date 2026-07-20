"""Run the physics-spine benchmark on the frozen shot set → a versioned YAML stamp.

Reuses the VALIDATED greens-filament-solver §2 gate solve path (the K=2 position
scaffold + rich non-negative ladder, both substrates) so the benchmark measures exactly
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
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET, SHOTSET_VERSION

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


def _fsa_roughness_sweep(fit, grid, bt0, *, n_p, n_f, nonneg) -> dict[int, float]:
    """d-roughness at each n_rho in the sweep, from a fit's force-balanced ψ."""
    from imas_ambix.latent.current_diffusion import flux_surface_geometry

    out: dict[int, float] = {}
    if not (fit.scored and fit.psi is not None and fit.coeffs is not None):
        return out
    if not (np.isfinite(fit.target[0]) and float(fit.target[0]) <= CONFINED_AXIS_R_MAX):
        return out
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
        )
        if geo is None:
            continue
        r = _d_roughness(np.asarray(geo.d_face))
        if np.isfinite(r):
            out[n_rho] = r
    return out


def _median(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


# --- the benchmark ----------------------------------------------------------


def run_stamp(
    *, created_utc: str, max_slices: int = 6, sigma: float = 0.02, shots=None
) -> SpineBenchmarkStamp:
    """Solve the frozen shot set under both substrates and assemble the stamp."""
    from imas_ambix.latent.boundary_disc import disc_read
    from imas_ambix.latent.gs_solve import (
        SUBSTRATE_GREENS,
        SUBSTRATE_GRID,
        EquilibriumGrid,
    )
    from scripts.greens_filament_gate_eval import _profile, _solve_arm
    from scripts.heldout_mse_gate_eval import _campaign_table, shot_bt0
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import factory_shot_payloads, frozen_spine_config

    shotset = shots if shots is not None else FROZEN_SHOTSET
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

        # per-substrate accumulators
        acc = {
            SUBSTRATE_GRID: {"dt": [], "fits": [], "conf": [], "conv": [], "rough": []},
            SUBSTRATE_GREENS: {
                "dt": [],
                "fits": [],
                "conf": [],
                "conv": [],
                "rough": [],
            },
        }
        grids = {SUBSTRATE_GRID: grid_gs, SUBSTRATE_GREENS: grid_free}
        for pos, k in enumerate(order):
            p = payload["payloads"][int(k)]
            inv = disc_read(p, grid_gs, tbl, basis)
            if inv is None or inv.ring is None:
                continue
            centroid = (float(inv.centroid_r), float(inv.centroid_z))
            disc_seed = _disc_seed_flat(grid_gs, inv)
            for sub in (SUBSTRATE_GRID, SUBSTRATE_GREENS):
                fit, dt = _solve_arm(
                    grids[sub],
                    sub,
                    None,
                    p,
                    centroid,
                    disc_seed,
                    tbl,
                    basis,
                    **spine_kw,
                )
                a = acc[sub]
                a["fits"].append((int(k), fit))
                if pos > 0:  # discard each shot's first slice as timing warmup
                    a["dt"].append(dt)
                if fit.scored:
                    a["conv"].append(1.0)
                    cr = float(fit.target[0]) if np.isfinite(fit.target[0]) else 1e9
                    a["conf"].append(1.0 if cr <= CONFINED_AXIS_R_MAX else 0.0)
                    rough = _fsa_roughness_sweep(
                        fit, grids[sub], bt0, n_p=n_p, n_f=n_f, nonneg=nonneg
                    )
                    if rough:
                        a["rough"].append(rough)
                else:
                    a["conv"].append(0.0)

        # reproduction (grid-free vs grid-GS) per slice
        gs_by_k = {k: f for k, f in acc[SUBSTRATE_GRID]["fits"]}
        fr_by_k = {k: f for k, f in acc[SUBSTRATE_GREENS]["fits"]}
        axis_cm, lcfs_cm, prof_rms = [], [], []
        from scripts.greens_filament_gate_eval import _axis_cm, _lcfs_cm, _profile_rms

        for k in gs_by_k:
            fg, ff = gs_by_k[k], fr_by_k.get(k)
            if ff is None or not (fg.scored and ff.scored):
                continue
            axis_cm.append(_axis_cm(fg.target, ff.target))
            lcfs_cm.append(_lcfs_cm(fg.target, ff.target))
            jg = _profile(fg, grid_gs, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
            jf = _profile(ff, grid_free, bt0, n_p=n_p, n_f=n_f, nonneg=nonneg)
            prof_rms.append(_profile_rms(jg, jf))

        for sub in (SUBSTRATE_GRID, SUBSTRATE_GREENS):
            a = acc[sub]
            n_att = len(a["fits"])
            n_scored = int(sum(a["conv"]))
            wall_ms = [d * 1e3 for d in a["dt"]]
            metrics: dict[str, float] = {}
            if wall_ms:
                metrics["solve_wall_ms_per_slice"] = float(np.median(wall_ms))
                metrics["throughput_slices_per_core_s"] = float(
                    1000.0 / np.median(wall_ms)
                )
            if a["conv"]:
                metrics["converged_fraction"] = float(np.mean(a["conv"]))
            if a["conf"]:
                metrics["confined_fraction"] = float(np.mean(a["conf"]))
            # FSA roughness: median over slices, per n_rho + slope
            r32 = _median([r.get(32, np.nan) for r in a["rough"]])
            r96 = _median([r.get(96, np.nan) for r in a["rough"]])
            if np.isfinite(r32):
                metrics["fsa_d_roughness_nrho32"] = r32
            if np.isfinite(r96):
                metrics["fsa_d_roughness_nrho96"] = r96
            # slope of median-roughness vs log2(n_rho)
            xs, ys = [], []
            for nr in _NRHO_SWEEP:
                med = _median([r.get(nr, np.nan) for r in a["rough"]])
                if np.isfinite(med):
                    xs.append(np.log2(nr))
                    ys.append(med)
            if len(xs) >= 2:
                metrics["fsa_d_roughness_resolution_slope"] = float(
                    np.polyfit(xs, ys, 1)[0]
                )
            # reproduction only meaningful on the grid-free arm (vs grid-GS ref)
            if sub == SUBSTRATE_GREENS:
                if axis_cm:
                    metrics["axis_reproduce_cm"] = _median(axis_cm)
                if lcfs_cm:
                    metrics["lcfs_reproduce_cm"] = _median(lcfs_cm)
                if prof_rms:
                    metrics["profile_reproduce_rms"] = _median(prof_rms)
            stamps.append(
                ShotStamp(
                    shot_id=shot,
                    role=bs.role,
                    campaign_signature=sig,
                    substrate=sub,
                    n_slices_attempted=n_att,
                    n_slices_scored=n_scored,
                    timing_n_repeat=len(wall_ms),
                    metrics=metrics,
                )
            )

    # aggregate rollups: median of each metric across shots, per substrate
    agg: dict[str, dict[str, float]] = {}
    for sub in {s.substrate for s in stamps}:
        keys = {k for s in stamps if s.substrate == sub for k in s.metrics}
        agg[sub] = {
            k: _median(
                [s.metrics[k] for s in stamps if s.substrate == sub and k in s.metrics]
            )
            for k in sorted(keys)
        }

    return SpineBenchmarkStamp(
        schema_version=SCHEMA_VERSION,
        shotset_version=SHOTSET_VERSION,
        created_utc=created_utc,
        machine=_machine_info(),
        env=_env_info("closure-spine-D2", spine_sha),
        shots=stamps,
        aggregate=agg,
        notes=[
            "Perf timing at OMP=1 (set OMP_NUM_THREADS=1); per-shot first slice "
            "discarded as warmup.",
            "Reproduction metrics (axis/lcfs/profile) are grid-free vs grid-GS on the "
            "same slice; recorded on the greens-matvec substrate row.",
            "FSA d-roughness = rms(2nd-diff of d_face)/rms(d_face), interior faces.",
            "RECONCILE: the greens-filament-solver §3 plan cites a cell-binned "
            "d-roughness ~0.5 worsening 0.45→0.72 with n_rho; measured here on the "
            "current CONTOUR-INTEGRATED geo.d_face the slope is ≤0 (improves with "
            "resolution) — no committed diagnostic backs the plan's number, so §3 "
            "should re-baseline roughness against this metric before claiming a fix.",
            "Held-out-MSE pitch is tracked separately (heldout_mse_gate_eval).",
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
