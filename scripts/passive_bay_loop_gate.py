#!/usr/bin/env python
"""Gate G-A: does the already-fitted vessel passive R reproduce the bay-loop
coupling deficit — the scorecard's −30–50% P4/P5 flux-loop "gain collapse" —
from first principles?

No new fitting, no 3D geometry.  We take the frozen vessel-resistance
calibration (``passive_resistance_calibration.json``, fitted 2026-07-17 on the
coil-only vacuum pool with the case currents held back) and ask the narrow
question the scorecard raised: on the held-out vacuum shots, is the bay flux
loops' apparent under-coupling explained by the modelled vessel eddy?

Mechanism.  On a coil-only ramp a bay loop measures ``Φ = G_coil·I_coil +
Φ_eddy``, where ``Φ_eddy`` is the field of the PF-ramp-induced vessel currents
(Lenz: anti-correlated with the drive).  A STATIC vacuum-gain regression that
omits the eddy attributes that anti-correlated flux to a REDUCED coil gain —
the scorecard's "collapse".  If the vessel model is right, adding its eddy
prediction restores the coil gain to ≈1 and explains the bay-loop eddy variance.

Metrics, per bay loop, pooled over the held-out shots (per-shot offset removed):
* eddy variance explained  = 1 − Var(meas_resid − Φ_eddy_model)/Var(meas_resid),
  where ``meas_resid`` = measured − static-coil model (the pure eddy signal);
* effective coil gain  = 1 + Cov(residual, static)/Var(static), evaluated
  BEFORE (residual = meas_resid → the collapsed gain) and AFTER (residual =
  meas_resid − Φ_eddy_model → should return to ≈1).
The centre-column control loops (``fl_cc*``, era-stable in the scorecard) must
stay ≈1 both before and after — the discriminator that the correction is
physical, not a global rescale.

The vessel/case resistances are FIXED machine hardware, so one calibration
applies across campaigns; per the per-shot sweep the era "drift" is a
drive-mixture shift over this fixed coupling, not an RMP-install step — so a
single vessel model reproducing the deficit IS the era-resolved claim.

Firewall: coil-only vacuum slices, raw amb magnetics + raw amc winding currents
+ geometry-only operator + the frozen resistance calibration; case currents are
held-back state (never a drive); NO EFIT, NO plasma.  Operator unchanged.

Artifact: imas_ambix/latent/artifacts/patch_gate/passive_bay_loop_gate.json
Figure:   docs/figures/nonaxisymmetric-field-subtraction/fig-passive-bay-loop-gate.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.passive_resistance import (
    campaign_mode_maps,
    load_calibration,
    zoh_mode_response,
)
from scripts.flux_loop_column_decomposition import BAY_LOOPS, select_cohort
from scripts.vacuum_passive_resistance_fit import _campaign_system, prep_shot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("passive_bay_loop_gate")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/nonaxisymmetric-field-subtraction")
CALIB = ARTIFACTS / "passive_resistance_calibration.json"
FIT = ARTIFACTS / "vacuum_passive_resistance_fit.json"

CONTROL_PREFIX = "fl_cc"  # centre-column loops: the era-stable far set


def _eddy_prediction(data, maps) -> np.ndarray:
    """Modelled eddy magnetics at every sensor, (T, S)."""
    a = zoh_mode_response(maps.tau, data.dt, data.psi_circ @ maps.v)
    return a @ maps.a_sens_modes.T


def _static_cols(table, system) -> tuple[np.ndarray, list[str]]:
    """g_pf columns for the system's drive channels, on the operator sensor axis."""
    fwd = build_operator(table)
    col_of = {ch: c for c, ch in enumerate(fwd.pf_amc_channels)}
    g = np.zeros((len(fwd.sensor_channels), len(system.coil_channels)))
    for k, ch in enumerate(system.coil_channels):
        c = col_of.get(ch, -1)
        if c >= 0:
            g[:, k] = fwd.g_pf[:, c]
    return g, list(fwd.sensor_channels)


def _pool_metrics(records: list[dict], channels: list[str], targets: list[str]) -> dict:
    """Per-channel pooled gain (before/after) + eddy variance explained."""
    ch_idx = {c: i for i, c in enumerate(channels)}
    out: dict[str, dict] = {}
    for ch in targets:
        if ch not in ch_idx:
            continue
        s = ch_idx[ch]
        stat, resid, eddy = [], [], []
        for r in records:
            st = r["static"][:, s]
            mr = r["meas_resid"][:, s]
            ed = r["eddy"][:, s]
            good = np.isfinite(st) & np.isfinite(mr) & np.isfinite(ed)
            if good.sum() < 50:
                continue
            # per-shot offset removal (matches the fit's intercept nuisance)
            st = st[good] - st[good].mean()
            mr = mr[good] - mr[good].mean()
            ed = ed[good] - ed[good].mean()
            stat.append(st)
            resid.append(mr)
            eddy.append(ed)
        if not stat:
            continue
        st = np.concatenate(stat)
        mr = np.concatenate(resid)
        ed = np.concatenate(eddy)
        vst = float(st @ st)
        gain_before = 1.0 + float(mr @ st) / vst if vst > 0 else float("nan")
        gain_after = 1.0 + float((mr - ed) @ st) / vst if vst > 0 else float("nan")
        vmr = float(mr @ mr)
        ev = 1.0 - float((mr - ed) @ (mr - ed)) / vmr if vmr > 0 else float("nan")
        # eddy signal size relative to the coil signal (why the gain moves)
        eddy_frac = float(np.sqrt(vmr / (vst + 1e-30)))
        out[ch] = {
            "gain_before": gain_before,
            "gain_after": gain_after,
            "eddy_var_explained": ev,
            "eddy_to_coil_rms": eddy_frac,
            "n_samples": int(st.size),
        }
    return out


def _family(metrics: dict, members: list[str]) -> dict | None:
    rows = [metrics[m] for m in members if m in metrics]
    if not rows:
        return None
    return {
        "n_channels": len(rows),
        "gain_before_median": float(np.median([r["gain_before"] for r in rows])),
        "gain_after_median": float(np.median([r["gain_after"] for r in rows])),
        "eddy_var_explained_median": float(
            np.median([r["eddy_var_explained"] for r in rows])
        ),
        "eddy_to_coil_rms_median": float(
            np.median([r["eddy_to_coil_rms"] for r in rows])
        ),
        "members": members,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--cohort", choices=("heldout", "sweep"), default="sweep",
        help="'sweep' = the P4/P5 coil-sweep vacuum cohort where the bay loops "
        "see their adjacent coils (the scorecard regime; a cross-regime held-out "
        "test since R was fit on CS preludes); 'heldout' = the vessel-fit's own "
        "held-out CS-precharge preludes.",
    )
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    cal = load_calibration(CALIB)
    logger.info("calibration level=%s groups=%d", cal.level, len(cal.group_multipliers))
    if args.cohort == "heldout":
        held = json.loads(FIT.read_text())["pool"]["held_out_shots"]
    else:
        held = select_cohort()
    if args.limit:
        held = held[: args.limit]
    logger.info("%s cohort: %d shots", args.cohort, len(held))

    # ---- prep held-out shots, group by campaign ----
    by_campaign: dict[str, list] = {}
    tables: dict[str, object] = {}
    for shot in held:
        rec = prep_shot((shot, "dedicated_vacuum", args.nr, args.nz))
        if rec is None:
            continue
        d = rec["data"]
        by_campaign.setdefault(d.campaign, []).append(d)
        if d.campaign not in tables:
            tables[d.campaign] = read_geometry_table(int(shot))
    logger.info("prepared %d shots over %d campaigns",
                sum(len(v) for v in by_campaign.values()), len(by_campaign))

    # ---- per campaign: tuned-R eddy prediction on the operator sensor axis ----
    records: list[dict] = []
    channels_ref: list[str] | None = None
    for key, shots in by_campaign.items():
        table = tables[key]
        system = _campaign_system(table, _grid(table, args.nr, args.nz))
        mult = cal.per_circuit(system.circuits, system.centroid_r, system.centroid_z)
        maps = campaign_mode_maps(system, mult)
        g_cols, channels = _static_cols(table, system)
        channels_ref = channels_ref or channels
        for d in shots:
            static = d.i_drive @ g_cols.T
            eddy = _eddy_prediction(d, maps)
            records.append(
                {"static": static, "meas_resid": d.meas_resid, "eddy": eddy}
            )

    assert channels_ref is not None
    bay = [b for b in BAY_LOOPS if b in channels_ref]
    controls = [c for c in channels_ref if c.startswith(CONTROL_PREFIX)]
    metrics = _pool_metrics(records, channels_ref, bay + controls)

    bay_fam = _family(metrics, bay)
    ctrl_fam = _family(metrics, controls)

    # ---- verdict ----
    gate_pass = bool(
        bay_fam
        and bay_fam["gain_before_median"] < 0.9  # a real deficit exists
        and abs(bay_fam["gain_after_median"] - 1.0) < abs(bay_fam["gain_before_median"] - 1.0) * 0.5
        and bay_fam["eddy_var_explained_median"] > 0.5
        and ctrl_fam
        and abs(ctrl_fam["gain_before_median"] - 1.0) < 0.1  # controls were fine
    )

    out = {
        "kind": "passive-bay-loop-gate-G-A",
        "leakage_free": True,
        "firewall": (
            "coil-only vacuum held-out shots; raw amb + raw amc winding drives + "
            "geometry-only operator + frozen resistance calibration; cases are "
            "held-back state; NO EFIT, NO plasma; operator unchanged."
        ),
        "calibration": {"level": cal.level, "source": str(CALIB)},
        "cohort": args.cohort,
        "shots_requested": held,
        "n_records": len(records),
        "bay_loops": bay,
        "controls": controls,
        "bay_family": bay_fam,
        "control_family": ctrl_fam,
        "per_channel": metrics,
        "gate_pass": gate_pass,
        "interpretation": (
            "gain_before < 1 = the scorecard bay-loop coupling deficit; "
            "gain_after → 1 with high eddy_var_explained = the modelled vessel "
            "eddy reproduces it from first principles (gate G-A); controls stay "
            "≈1 = the correction is physical, not a global rescale."
        ),
    }
    (ARTIFACTS / "passive_bay_loop_gate.json").write_text(json.dumps(out, indent=2))
    logger.info("bay family: %s", bay_fam)
    logger.info("control family: %s", ctrl_fam)
    logger.info("GATE G-A pass=%s", gate_pass)

    _figure(metrics, bay, controls, bay_fam, ctrl_fam, gate_pass)
    return 0


def _grid(table, nr, nz):
    from imas_ambix.latent.gs_solve import EquilibriumGrid  # noqa: PLC0415

    return EquilibriumGrid.from_table(table, nr=nr, nz=nz)


def _figure(metrics, bay, controls, bay_fam, ctrl_fam, gate_pass) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, group, title, color in (
        (ax[0], bay, "bay loops (P4/P5)", "#b00"),
        (ax[1], controls, "centre-column controls (fl_cc*)", "#1565c0"),
    ):
        names = [c for c in group if c in metrics]
        xs = np.arange(len(names))
        gb = [metrics[c]["gain_before"] for c in names]
        ga = [metrics[c]["gain_after"] for c in names]
        w = 0.4
        a.bar(xs - w / 2, gb, w, color=color, alpha=0.55, label="gain before (no eddy)")
        a.bar(xs + w / 2, ga, w, color=color, alpha=1.0, label="gain after (tuned vessel R)")
        a.axhline(1.0, color="k", lw=1.0)
        a.set_xticks(xs)
        a.set_xticklabels(names, rotation=60, fontsize=7)
        a.set_ylabel("effective coil gain (empirical/model)")
        a.set_ylim(0, 1.4)
        a.set_title(title)
        a.legend(fontsize=8)
    verdict = "PASS" if gate_pass else "PARTIAL"
    sub = (
        f"bay gain {bay_fam['gain_before_median']:.2f}→{bay_fam['gain_after_median']:.2f}, "
        f"eddy var expl {bay_fam['eddy_var_explained_median']:.2f}; "
        f"controls {ctrl_fam['gain_before_median']:.2f}→{ctrl_fam['gain_after_median']:.2f}"
        if bay_fam and ctrl_fam else ""
    )
    fig.suptitle(f"Gate G-A — vessel eddy reproduces the bay-loop coupling deficit "
                 f"[{verdict}]\n{sub}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig-passive-bay-loop-gate.png", dpi=140)
    plt.close(fig)
    logger.info("wrote figure")


if __name__ == "__main__":
    sys.exit(main())
