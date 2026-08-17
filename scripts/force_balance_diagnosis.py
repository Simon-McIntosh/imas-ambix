#!/usr/bin/env python
"""Measured-state force-balance diagnosis of the forward operator.

Answers ONE question with classical physics only: does the vacuum field the
forward operator predicts AT THE PLASMA satisfy the vertical field a confined
equilibrium REQUIRES there?  The vacuum-shot campaign validated the coil→sensor
map at 0.1–0.44σ, yet the operator has no confined fixed point on measured
coil currents — so the error is either in the field where the sensors do not
sample (at the plasma) or solve-side.  This sweep separates them; no fitting,
no learned components, no solver changes.

Per shot × slice (measured coil currents, vessel eddies predicted from the
coil drives by the passive circuit system, machine-quiescent start):

* model vacuum B_z(R) along the axis midplane through the SAME finite-area
  cylinder kernel and per-coil merge the sensor operator uses
  (:mod:`imas_ambix.gs.force_balance`);
* the Shafranov requirement B_v at the measured axis, parameterised by the
  firewalled EFIT referee quantities (R_axis, a, βp, li) — DIAGNOSTIC-ONLY:
  they enter an analytic identity, nothing tunes on them;
* the vacuum decay index n = −(R/B_z)·∂B_z/∂R at the axis against the rigid
  stability window 0 < n < 3/2;
* a per-coil-group waterfall of B_z at the axis (sol / p2 / p3 / p4 / p5 /
  p6 / case / vessel) — the localizer that names a suspect;
* optionally, the reconstruction-mode probe: the frozen classical spine
  (measurement-constrained interior solve) run on the same shot's raw
  magnetics — if the vacuum field checks out AND the measurement-constrained
  solve still walks the axis outboard, the culprit is solve-side by
  elimination.

Branch rule:
  Coil-side:  flat-top |B_z,model − B_v,req|/|B_v,req| > 0.15 or a sign
              error, AND the waterfall localizes ≥ 70% of the field at
              the axis to ≤ 2 coil groups.
  Solve-side: B_z discrepancy below 5%, decay index inside the window, yet the
              measurement-constrained solve deconfines.
  Ambiguous:  discrepancy 5–15% or a non-localizing waterfall → extend once
              by +3 shots; never pick a branch by judgement.

Artifact: imas_ambix/gs/artifacts/force_balance_diagnosis.json
Figures:  docs/figures/equilibrium-realism/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.gs import force_balance as fb
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import feature_schema, load_shot_slices_raw
from imas_ambix.latent.gs_solve import EquilibriumGrid
from imas_ambix.latent.temporal_operator import (
    build_passive_circuit_system,
    predict_vessel_currents,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("force_balance_diagnosis")

ARTIFACT = Path("imas_ambix/gs/artifacts/force_balance_diagnosis.json")
FIGURES = Path("docs/figures/equilibrium-realism")

#: campaign anchor + flat-top-rich + ramp-heavy train-split shots (rule: the
#: anchor, the lowest 10–90 ramp fraction, and the highest-peak ramp-dominated
#: shot among the leading train-split shots with L2 equilibrium coverage)
DEFAULT_SHOTS = (11766, 11772, 11767)
R_LINE = np.linspace(0.4, 1.4, 201)
IP_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GROUP_ORDER = ("sol", "p2", "p3", "p4", "p5", "p6", "case", "vessel")


def _read_referee(shot: int) -> dict[str, np.ndarray]:
    """L2 equilibrium referee quantities — DIAGNOSTIC-ONLY (locked decision)."""
    import zarr  # noqa: PLC0415

    from imas_ambix.worldmodel.equilibrium_labels import (  # noqa: PLC0415
        equilibrium_store_path,
    )

    eq = zarr.open_group(str(equilibrium_store_path(shot, None)), mode="r")[
        "equilibrium"
    ]
    return {
        k: np.asarray(eq[k], dtype=np.float64)
        for k in (
            "time",
            "magnetic_axis_r",
            "magnetic_axis_z",
            "minor_radius",
            "elongation",
            "beta_pol",
            "li",
        )
    }


def _interp_ref(ref: dict[str, np.ndarray], key: str, t: float) -> float:
    tt = ref["time"]
    yy = ref[key]
    ok = np.isfinite(tt) & np.isfinite(yy)
    if ok.sum() < 2 or t < tt[ok][0] or t > tt[ok][-1]:
        return float("nan")
    return float(np.interp(t, tt[ok], yy[ok]))


def _vessel_currents(
    table, grid, i_pf_full, channels, times, ip_amperes=None, axis_rz=None
):
    """Vessel circuit currents from the measured drives, quiescent start.

    Thin wrapper over :func:`imas_ambix.latent.temporal_operator.
    predict_vessel_currents` (the drive mechanism now lives in the forward
    chain, not in this diagnosis script): exact-ZOH eigenmode integration of
    the passive circuit system driven by the full measured coil history AND —
    when ``ip_amperes``/``axis_rz`` are given — by the plasma current's own
    flux swing (a filament at the referee axis trace), whose
    Lenz-anti-parallel image currents contribute confining vertical field
    during the fast Ip ramp.  Returns ``(circuits, i_vessel_coil (T, P),
    i_vessel_full (T, P))``.
    """
    vsys = build_passive_circuit_system(table, grid)
    i_coil, i_full = predict_vessel_currents(
        table,
        vsys,
        i_pf_full,
        channels,
        times,
        ip_amperes=ip_amperes,
        axis_rz=axis_rz,
    )
    return vsys.circuits, i_coil, i_full


def _select_slices(
    ip_ka: np.ndarray, plasma_on: np.ndarray, covered: np.ndarray
) -> list[tuple[int, str]]:
    """Rising-side slices at IP_FRACTIONS of peak + first/mid/last top slices.

    ``covered`` marks slices inside the referee's time coverage — the Shafranov
    identity cannot be parameterised outside it, so the peak is defined over
    the COVERED plasma-on window and out-of-coverage picks are dropped LOUDLY
    (the early fast ramp routinely precedes the first EFIT slice).
    """
    ip = np.abs(np.where(np.isfinite(ip_ka), ip_ka, 0.0))
    usable = plasma_on & covered
    if not usable.any():
        return []
    peak = float(ip[usable].max())
    k_peak = int(np.nonzero(usable & (ip >= peak))[0][0])
    picks: list[tuple[int, str]] = []
    for f in IP_FRACTIONS:
        rising = np.nonzero((ip[: k_peak + 1] >= f * peak) & plasma_on[: k_peak + 1])[0]
        if not rising.size:
            continue
        k = int(rising[0])
        if covered[k]:
            picks.append((k, f"ramp{f:.1f}"))
        else:
            logger.info("  dropped ramp%.1f pick (t outside referee coverage)", f)
    flat = np.nonzero((ip >= 0.9 * peak) & usable)[0]
    if flat.size:
        for k, tag in (
            (flat[0], "flat_first"),
            (flat[flat.size // 2], "flat_mid"),
            (flat[-1], "flat_last"),
        ):
            picks.append((int(k), tag))
    seen: set[int] = set()
    out = []
    for k, tag in picks:
        if k not in seen:
            seen.add(k)
            out.append((k, tag))
    return out


def diagnose_shot(shot: int) -> dict | None:
    """The measured-state sweep for one shot."""
    schema = feature_schema()
    table = read_geometry_table(shot)
    fwd = build_operator(table)
    loaded = load_shot_slices_raw(shot, schema)
    if loaded is None:
        logger.warning("shot %d: no level-1 data", shot)
        return None
    x, times, plasma_on = loaded
    ref = _read_referee(shot)

    from imas_ambix.latent.data import anchored_columns, schema_group_offsets

    offsets = schema_group_offsets(schema)
    amc_names = schema["amc"]
    ip_col, _ne = anchored_columns(schema)
    amc_block = x[:, offsets["amc"] : offsets["amc"] + len(amc_names)]
    n_t = x.shape[0]
    i_pf = np.zeros((n_t, len(fwd.pf_amc_channels)))
    for t in range(n_t):
        vals = {
            ch: float(amc_block[t, j])
            for j, ch in enumerate(amc_names)
            if np.isfinite(amc_block[t, j])
        }
        i_pf[t] = fwd.assemble_pf_currents(vals)

    ip_ka = x[:, ip_col]
    tt = ref["time"]
    ok = np.isfinite(tt) & np.isfinite(ref["magnetic_axis_r"])
    covered = (
        (times >= tt[ok][0]) & (times <= tt[ok][-1])
        if ok.sum() >= 2
        else np.zeros_like(plasma_on)
    )

    # plasma filament trace for the vessel drive: measured Ip at the referee
    # axis (np.interp clamps outside coverage — the early-ramp linkage is then
    # approximate, but those eddies decay on the vessel τ ≤ 35 ms)
    ip_amp_t = np.where(np.isfinite(ip_ka), ip_ka, 0.0) * 1e3
    okz = np.isfinite(tt) & np.isfinite(ref["magnetic_axis_z"])
    axis_rz = np.column_stack(
        [
            np.interp(times, tt[ok], ref["magnetic_axis_r"][ok]),
            np.interp(times, tt[okz], ref["magnetic_axis_z"][okz]),
        ]
    )
    grid = EquilibriumGrid.from_table(table, nr=65, nz=97)
    circuits, i_vessel, i_vessel_full = _vessel_currents(
        table, grid, i_pf, fwd.pf_amc_channels, times, ip_amp_t, axis_rz
    )
    picks = _select_slices(ip_ka, plasma_on, covered)
    usable = plasma_on & covered
    peak_ka = float(np.nanmax(np.abs(ip_ka[usable]))) if usable.any() else float("nan")

    rows = []
    profiles = []
    for k, tag in picks:
        t = float(times[k])
        r_ax = _interp_ref(ref, "magnetic_axis_r", t)
        z_ax = _interp_ref(ref, "magnetic_axis_z", t)
        a_min = _interp_ref(ref, "minor_radius", t)
        kappa = _interp_ref(ref, "elongation", t)
        betap = _interp_ref(ref, "beta_pol", t)
        li = _interp_ref(ref, "li", t)
        ip_a = float(ip_ka[k]) * 1e3
        if not (np.isfinite(r_ax) and np.isfinite(a_min)):
            continue

        z_line = np.full_like(R_LINE, z_ax if np.isfinite(z_ax) else 0.0)
        channels, cols_line = fb.known_coil_bz(table, R_LINE, z_line)
        pas_line = fb.passive_circuit_bz(table, circuits, R_LINE, z_line)
        bz_line = cols_line @ i_pf[k] + pas_line @ i_vessel[k]
        n_line = fb.decay_index(R_LINE, bz_line)

        j_ax = int(np.argmin(np.abs(R_LINE - r_ax)))
        contrib = cols_line[j_ax] * i_pf[k]
        groups: dict[str, float] = dict.fromkeys(GROUP_ORDER, 0.0)
        for chan, c in zip(channels, contrib, strict=True):
            g = fb.coil_group(chan)
            groups[g] = groups.get(g, 0.0) + float(c)
        groups["vessel"] = float(pas_line[j_ax] @ i_vessel[k])
        bz_axis = float(sum(groups.values()))
        # the plasma-driven eddy share, reported ALONGSIDE the gate columns —
        # the pre-declared gates read rel_discrepancy (coil + coil-driven
        # eddies) unchanged; this column quantifies the missing-drive term
        bz_eddy_plasma = float(pas_line[j_ax] @ (i_vessel_full[k] - i_vessel[k]))

        betap_li2 = (
            betap + 0.5 * li
            if np.isfinite(betap) and np.isfinite(li)
            else (float("nan"))
        )
        bv_req = fb.shafranov_vertical_field(ip_a, r_ax, a_min, betap_li2)
        rel = (bz_axis - bv_req) / abs(bv_req) if np.isfinite(bv_req) else float("nan")
        # the identity's own uncertainty bracket: the elongation-corrected
        # requirement ln(8R/(a√κ)) is the soft edge of the band the circular
        # form anchors — a residual must clear BOTH before it can convict
        # the coil model (diagnostic columns; gate inputs unchanged)
        bv_req_elong = fb.shafranov_vertical_field_elongated(
            ip_a, r_ax, a_min, kappa, betap_li2
        )

        def _rel(bz: float, bv: float) -> float:
            return (bz - bv) / abs(bv) if np.isfinite(bv) else float("nan")

        mags = np.array([abs(groups[g]) for g in GROUP_ORDER])
        top2 = (
            float(np.sort(mags)[-2:].sum() / mags.sum()) if mags.sum() else float("nan")
        )
        top2_names = [GROUP_ORDER[i] for i in np.argsort(mags)[-2:][::-1]]

        rows.append(
            {
                "shot": shot,
                "t_index": int(k),
                "time_s": t,
                "tag": tag,
                "ip_amperes": ip_a,
                "ip_frac": float(abs(ip_ka[k]) / peak_ka),
                "r_axis_ref": r_ax,
                "z_axis_ref": z_ax,
                "minor_radius_ref": a_min,
                "kappa_ref": kappa,
                "betap_ref": betap,
                "li_ref": li,
                "betap_li2": betap_li2,
                "bz_model_axis": bz_axis,
                "bv_required": bv_req,
                "bv_required_elong": bv_req_elong,
                "rel_discrepancy": rel,
                "rel_discrepancy_elong": _rel(bz_axis, bv_req_elong),
                "bz_eddy_plasma": bz_eddy_plasma,
                "rel_discrepancy_with_plasma_eddies": (
                    (bz_axis + bz_eddy_plasma - bv_req) / abs(bv_req)
                    if np.isfinite(bv_req)
                    else float("nan")
                ),
                "rel_discrepancy_elong_with_plasma_eddies": _rel(
                    bz_axis + bz_eddy_plasma, bv_req_elong
                ),
                "sign_error": bool(np.isfinite(bv_req) and bz_axis * bv_req < 0.0),
                "decay_index_axis": float(n_line[j_ax]),
                "waterfall": {g: float(groups[g]) for g in GROUP_ORDER},
                "top2_share": top2,
                "top2_groups": top2_names,
            }
        )
        profiles.append(
            {
                "shot": shot,
                "tag": tag,
                "ip_frac": rows[-1]["ip_frac"],
                "bz_line": bz_line.tolist(),
                "n_line": n_line.tolist(),
                "r_axis_ref": r_ax,
                "bv_required": bv_req,
            }
        )
    return {"shot": shot, "peak_ip_ka": peak_ka, "rows": rows, "profiles": profiles}


def probe_reconstruction(shot: int, max_slices: int) -> list[dict]:
    """Reconstruction-mode axis-hold probe: the frozen spine on raw magnetics.

    Measurement-constrained interior solve (frozen physics spine, byte-same config
    as the label factory) — records whether the solve holds the axis where
    the data puts it or walks it toward the outboard attractor.  EFIT axis is
    read AFTERWARDS for the trace comparison (diagnostic-only).
    """
    from scripts.closure_gate_eval import (  # noqa: PLC0415
        _shot_passive_sidecar,
        fit_and_read_slice,
    )
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, _sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    spc = dict(spine["soft_priors"])
    spc["boundary_prior"] = spc.pop("boundary_prior", "disc")
    payload = factory_shot_payloads(
        shot, nr=65, nz=97, max_slices=max_slices, min_ip_ka=60.0
    )
    if payload is None:
        return []
    grid, table = payload["grid"], payload["table"]
    sidecar = _shot_passive_sidecar(payload, int(isolve["passive_k"]))
    ref = _read_referee(shot)

    out = []
    warm = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    for kk in order:
        p = payload["payloads"][int(kk)]
        f = fit_and_read_slice(
            grid,
            table,
            p,
            beta0_grid=(0.5,),
            alpha_grid=(1.0,),
            cost_limit=float("inf"),
            convergence_limit=5e-3,
            retry_max_iterations=160,
            fit_mode="ladder",
            n_p=int(isolve["n_p"]),
            n_f=int(isolve["n_f"]),
            smoothness=float(isolve["smoothness"]),
            nonneg=isolve["profile_kind"] == "monomial-nonneg",
            passive=sidecar,
            passive_ridge=1.0,
            warm_jphi=warm,
            reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
            keep_psi=True,
            keep_jphi=True,
            basis=payload["basis"],
            meta={},
            soft_prior_cfg=spc,
            boundary_read=isolve["boundary_read_scoring"],
        )
        row = {
            "shot": shot,
            "time_s": p.time_s,
            "ip_amperes": p.ip_amperes,
            "scored": bool(f.scored),
            "converged": bool(getattr(f, "converged", False)),
            "axis_r": float("nan"),
            "axis_z": float("nan"),
            "axis_r_ref": _interp_ref(ref, "magnetic_axis_r", p.time_s),
            "axis_z_ref": _interp_ref(ref, "magnetic_axis_z", p.time_s),
        }
        if f.scored and f.target is not None:
            row["axis_r"] = float(f.target[0])
            row["axis_z"] = float(f.target[1])
        if f.scored and f.converged and f.jphi_flat is not None:
            warm = f.jphi_flat
        out.append(row)
        logger.info(
            "probe %d t=%.3f axis %.3f (ref %.3f) conv=%s",
            shot,
            p.time_s,
            row["axis_r"],
            row["axis_r_ref"],
            row["converged"],
        )
    return out


def evaluate_gates(shot_results: list[dict], probe_rows: list[dict]) -> dict:
    """Evaluate the coil-side, solve-side, and ambiguity branch rule."""
    flat = [
        r
        for s in shot_results
        for r in s["rows"]
        if r["tag"].startswith("flat") and np.isfinite(r["rel_discrepancy"])
    ]
    rel = np.array([r["rel_discrepancy"] for r in flat])
    sign = any(r["sign_error"] for r in flat)
    top2 = np.array([r["top2_share"] for r in flat])
    n_ax = np.array([r["decay_index_axis"] for r in flat])
    lo, hi = fb.DECAY_INDEX_WINDOW

    med_abs_rel = float(np.median(np.abs(rel))) if rel.size else float("nan")
    localizes = bool(np.median(top2) >= 0.70) if top2.size else False
    n_in_window = (
        bool(np.median((n_ax > lo) & (n_ax < hi)) >= 0.5) if n_ax.size else False
    )
    probe_scored = [
        r
        for r in probe_rows
        if r["scored"]
        and np.isfinite(r["axis_r"])
        and np.isfinite(r["axis_r_ref"])  # no referee sample → not comparable
    ]
    probe_err = (
        float(np.median([abs(r["axis_r"] - r["axis_r_ref"]) for r in probe_scored]))
        if probe_scored
        else float("nan")
    )
    probe_deconfines = (
        bool(np.median([r["axis_r"] for r in probe_scored]) > 1.4)
        if probe_scored
        else None
    )

    # A 5-15% discrepancy neither clears nor convicts the coil model.  The
    # solve-side branch therefore requires a residual below the band floor and
    # an independent witness: the measurement-constrained probe still deconfines.
    coil_side = (med_abs_rel > 0.15 or sign) and localizes
    solve_side = (
        med_abs_rel < 0.05
        and not sign
        and n_in_window
        and probe_deconfines is True
    )
    verdict = "ambiguous"
    if coil_side:
        verdict = "coil-side"
    elif solve_side:
        verdict = "solve-side"
    ramp = [
        r
        for s in shot_results
        for r in s["rows"]
        if r["tag"].startswith("ramp") and np.isfinite(r["rel_discrepancy"])
    ]

    def _med(rows_, key):
        vals = [r[key] for r in rows_ if np.isfinite(r.get(key, float("nan")))]
        return float(np.median(vals)) if vals else float("nan")

    # identity-robustness band (diagnostic — the pre-declared gate inputs
    # above are untouched): the large-aspect identity's own uncertainty is
    # bracketed by its circular and elongation-corrected forms; the
    # eddy-folded flat-top residual is "inside the band" when the two forms
    # straddle zero (the true requirement lies between them) or when the
    # elongation-corrected residual sits within the ±5% inner band.
    med_flat_folded = _med(flat, "rel_discrepancy_with_plasma_eddies")
    med_flat_elong_folded = _med(flat, "rel_discrepancy_elong_with_plasma_eddies")
    inside_band = bool(
        np.isfinite(med_flat_folded)
        and np.isfinite(med_flat_elong_folded)
        and (
            med_flat_folded * med_flat_elong_folded <= 0.0
            or abs(med_flat_elong_folded) <= 0.05
        )
    )

    return {
        "identity_band": {
            "median_rel_flat_elong": _med(flat, "rel_discrepancy_elong"),
            "median_rel_flat_elong_with_plasma_eddies": med_flat_elong_folded,
            "median_rel_ramp_elong_with_plasma_eddies": _med(
                ramp, "rel_discrepancy_elong_with_plasma_eddies"
            ),
            "rule": (
                "inside iff the eddy-folded flat-top residual changes sign "
                "between the circular and elongation-corrected requirement "
                "forms, or the elongation-corrected residual is within the "
                "5% inner band"
            ),
            "flat_residual_inside_band": inside_band,
        },
        "flat_top_slices": len(flat),
        "median_abs_rel_discrepancy": med_abs_rel,
        # reported diagnostics (NOT gate inputs): the missing plasma-driven
        # vessel-eddy term folded in
        "median_rel_ramp": _med(ramp, "rel_discrepancy"),
        "median_rel_ramp_with_plasma_eddies": _med(
            ramp, "rel_discrepancy_with_plasma_eddies"
        ),
        "median_rel_flat_with_plasma_eddies": _med(
            flat, "rel_discrepancy_with_plasma_eddies"
        ),
        "sign_error_any": sign,
        "median_top2_share": float(np.median(top2)) if top2.size else float("nan"),
        "waterfall_localizes": localizes,
        "decay_index_in_window": n_in_window,
        "probe_axis_err_median_m": probe_err,
        "probe_deconfines": probe_deconfines,
        "coil_side_gate": bool(coil_side),
        "solve_side_gate": bool(solve_side),
        "verdict": verdict,
    }


def make_figures(shot_results: list[dict], probe_rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")

    # (1) B_z(R) overlays vs the Shafranov requirement
    fig, axes = plt.subplots(
        1, len(shot_results), figsize=(5.2 * len(shot_results), 4.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, shot_results, strict=True):
        for p in s["profiles"]:
            c = cmap(p["ip_frac"])
            ax.plot(R_LINE, p["bz_line"], color=c, lw=1.1)
            if np.isfinite(p["bv_required"]):
                ax.plot(p["r_axis_ref"], p["bv_required"], "x", color=c, ms=7, mew=1.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("R [m]")
        ax.set_title(f"shot {s['shot']}")
    axes[0].set_ylabel("vacuum $B_z$ [T]  (lines: model; ×: Shafranov req.)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, label="Ip fraction", shrink=0.85)
    fig.savefig(FIGURES / "fig-bz-overlays.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (2) relative discrepancy + decay index vs Ip fraction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    marker = {"11766": "o", "11772": "s", "11767": "^"}
    for si, s in enumerate(shot_results):
        rr = [r for r in s["rows"] if np.isfinite(r["rel_discrepancy"])]
        ax1.plot(
            [r["ip_frac"] for r in rr],
            [r["rel_discrepancy"] for r in rr],
            marker.get(str(s["shot"]), "o"),
            ls="-",
            ms=5,
            label=f"shot {s['shot']}",
        )
        ax1.plot(
            [r["ip_frac"] for r in rr],
            [r["rel_discrepancy_with_plasma_eddies"] for r in rr],
            marker.get(str(s["shot"]), "o"),
            ls="--",
            ms=5,
            mfc="none",
            color=f"C{si}",
            label="+ plasma-driven eddies" if si == 0 else None,
        )
        ax2.plot(
            [r["ip_frac"] for r in s["rows"]],
            [r["decay_index_axis"] for r in s["rows"]],
            marker.get(str(s["shot"]), "o"),
            ls="-",
            ms=5,
            label=f"shot {s['shot']}",
        )
    ax1.axhspan(
        -0.15,
        0.15,
        color="#4477aa",
        alpha=0.12,
        label="ambiguity window ±15%",
    )
    ax1.axhspan(-0.05, 0.05, color="#4477aa", alpha=0.18)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("Ip fraction")
    ax1.set_ylabel("(B$_z$ model − B$_v$ req) / |B$_v$ req|")
    ax1.legend(fontsize=8)
    lo, hi = fb.DECAY_INDEX_WINDOW
    ax2.axhspan(lo, hi, color="#228833", alpha=0.15, label="stability window")
    ax2.set_xlabel("Ip fraction")
    ax2.set_ylabel("decay index n at axis")
    ax2.legend(fontsize=8)
    fig.savefig(FIGURES / "fig-decay-index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (2b) identity-robustness band: the eddy-folded residual against the
    # circular vs elongation-corrected requirement forms, flat-top slices
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for si, s in enumerate(shot_results):
        flat_rows = [
            r
            for r in s["rows"]
            if r["tag"].startswith("flat")
            and np.isfinite(r.get("rel_discrepancy_elong", float("nan")))
        ]
        xs = np.full(len(flat_rows), si)
        circ = [r["rel_discrepancy_with_plasma_eddies"] for r in flat_rows]
        elong = [r["rel_discrepancy_elong_with_plasma_eddies"] for r in flat_rows]
        ax.plot(
            xs - 0.12,
            circ,
            "o",
            ms=6,
            color="#4477aa",
            label="vs circular ln(8R/a)" if si == 0 else None,
        )
        ax.plot(
            xs + 0.12,
            elong,
            "s",
            ms=6,
            color="#cc3311",
            label="vs elongation-corrected ln(8R/(a√κ))" if si == 0 else None,
        )
        for x, c, e in zip(xs, circ, elong, strict=True):
            ax.plot([x - 0.12, x + 0.12], [c, e], "-", lw=0.7, color="0.6")
    ax.axhspan(-0.05, 0.05, color="#228833", alpha=0.15, label="±5% inner band")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(shot_results)), [str(s["shot"]) for s in shot_results])
    ax.set_ylabel("flat-top (B$_z$+eddy − B$_v$ req) / |B$_v$ req|")
    ax.set_title("identity-robustness band — the residual the identity itself moves")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "fig-identity-band.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (3) per-coil-group waterfall at the measured axis (flat-top slices)
    fig, axes = plt.subplots(
        1, len(shot_results), figsize=(5.2 * len(shot_results), 4.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, shot_results, strict=True):
        flat = [r for r in s["rows"] if r["tag"] == "flat_mid"] or s["rows"][-1:]
        r = flat[0]
        vals = [r["waterfall"][g] for g in GROUP_ORDER]
        colors = ["#cc3311" if v > 0 else "#4477aa" for v in vals]
        ax.bar(range(len(GROUP_ORDER)), vals, color=colors)
        if np.isfinite(r["bv_required"]):
            ax.axhline(
                r["bv_required"], color="k", ls="--", lw=1.2, label="Shafranov req."
            )
        ax.axhline(r["bz_model_axis"], color="#228833", ls=":", lw=1.2, label="model")
        ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=45)
        ax.set_title(
            f"shot {s['shot']} @ {r['tag']} (Ip {r['ip_amperes'] / 1e3:.0f} kA)"
        )
        ax.legend(fontsize=8)
    axes[0].set_ylabel("$B_z$ at measured axis [T]")
    fig.savefig(FIGURES / "fig-waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (4) reconstruction-mode axis-hold trace
    if probe_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        t = [r["time_s"] for r in probe_rows]
        ax.plot(
            t,
            [r["axis_r"] for r in probe_rows],
            "o-",
            ms=5,
            color="#cc3311",
            label="measurement-constrained solve",
        )
        ax.plot(
            t,
            [r["axis_r_ref"] for r in probe_rows],
            "s--",
            ms=4,
            color="#4477aa",
            label="EFIT axis (diagnostic-only)",
        )
        ax.axhline(1.4, color="k", ls=":", lw=1, label="outboard attractor read")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("axis R [m]")
        ax.set_title(f"axis-hold probe — shot {probe_rows[0]['shot']}")
        ax.legend(fontsize=8)
        fig.savefig(FIGURES / "fig-axis-hold.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    ap.add_argument("--probe-shot", type=int, default=11766)
    ap.add_argument("--probe-slices", type=int, default=10)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    args = ap.parse_args()

    shot_results = []
    for s in args.shots:
        r = diagnose_shot(int(s))
        if r is not None:
            shot_results.append(r)
            for row in r["rows"]:
                logger.info(
                    "%d %s Ip=%.0fkA rel=%+.3f rel+eddy=%+.3f n=%.2f top2=%.2f (%s)",
                    row["shot"],
                    row["tag"],
                    row["ip_amperes"] / 1e3,
                    row["rel_discrepancy"],
                    row["rel_discrepancy_with_plasma_eddies"],
                    row["decay_index_axis"],
                    row["top2_share"],
                    "+".join(row["top2_groups"]),
                )

    probe_rows = (
        probe_reconstruction(int(args.probe_shot), int(args.probe_slices))
        if not args.no_probe
        else []
    )
    gates = evaluate_gates(shot_results, probe_rows)
    logger.info("GATES: %s", json.dumps(gates, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "shots": shot_results,
                "probe": probe_rows,
                "gates": gates,
                "r_line": R_LINE.tolist(),
                "localization_metric": (
                    "top-2 coil-group share of Σ|per-group B_z| at the axis"
                ),
            },
            indent=1,
        )
    )
    make_figures(shot_results, probe_rows)
    logger.info("artifact: %s; figures: %s", args.out, FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
