"""Raw-held-out prediction bar — the STRONG tier of the model-independent eval.

This is the bar a world model's raw-held-out prediction must beat: predict
withheld RAW observables the model never saw (the locked, leakage-audited MSE
held-out split; pitch on beam-on slices) and compare against two reference
predictors,

  * **persistence** — freeze the pitch profile at the first beam-on slice
    (cheap; no solver; the sanity floor), and
  * **EnKF** — the classical parameter-space ensemble smoother over TORAX
    current diffusion (the credible classical comparator).

Both reference predictors are scored on the SAME raw-held-out MSE harness
(:mod:`imas_ambix.statespace.mse_eval`) using the SAME shared forward/inverse
observation models, so the world model, persistence, and EnKF are all measured
on one ruler.  Nothing here re-implements scoring physics — this module is a
thin, importable composition of the existing D1 harness:

  * :mod:`imas_ambix.statespace.mse_eval`   — the scoring contract + metrics.
  * :mod:`imas_ambix.statespace.mse_split`  — the locked held-out split builder.
  * :mod:`imas_ambix.statespace.enkf_baseline` — the classical EnKF comparator.

The two public entry points:

  * :func:`persistence_bar` — score the (cheap) persistence reference on a
    manifest.  No solver, no /work-heavy reads beyond the manifest's truth.
  * :func:`prediction_bar` — assemble the full {persistence, EnKF} bar.  The
    EnKF leg is heavy (TORAX-referenced ensemble per shot); when it cannot run
    in-session it is filled from the locked reference artifact and flagged
    ``"from_reference"`` so the caller knows which legs were measured live.

The returned :class:`Bar` carries, per predictor, the pre-registered metrics
the harness produces: PRIMARY pitch RMSE / CRPS / NLL / cov90 (the gate axis),
the by-window (quiescent / transient) split, and the SECONDARY method-matched
q0 / rax blocks.  This is exactly the structure
:func:`imas_ambix.statespace.mse_eval.score` returns, hoisted to a flat,
comparable bar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_ambix.statespace import mse_eval

if TYPE_CHECKING:
    from imas_ambix.statespace.enkf_baseline import EnKFConfig

logger = logging.getLogger(__name__)

# Pre-registered acceptance / reference targets (from sequential-current-da-v1).
# The bar, run on the D1 split, should REPRODUCE these — they are documented
# here as the reference the live measure is checked against, NOT a tuning knob.
REFERENCE = {
    "enkf_pitch_rmse": 0.225,
    "enkf_pitch_rmse_ci": (0.199, 0.259),
    "persistence_pitch_rmse": 0.719,
    "physics_frontier_pitch_rmse": 0.19,  # pre-registered sightline floor
    "near_axis_rad_floor": 0.41,  # pre-registered near-axis interior floor
    "coverage_gate": (mse_eval.COVERAGE_GATE_LO, mse_eval.COVERAGE_GATE_HI),
}

# The locked reference EnKF metrics artifact (the 112-shot D1 bar).  Used to
# fill the EnKF leg when the heavy live run cannot complete in-session.
_REFERENCE_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "statespace"
    / "artifacts"
    / "enkf_baseline_metrics_v0.json"
)


# ---------------------------------------------------------------------------
# Bar structure
# ---------------------------------------------------------------------------


@dataclass
class BarLeg:
    """One predictor's scored block + provenance.

    ``source`` is ``"live"`` (scored in this process on the supplied manifest)
    or ``"from_reference"`` (read from the locked reference artifact because the
    heavy live run was not performed).  ``scored`` is the nested metric dict the
    harness :func:`~imas_ambix.statespace.mse_eval.score` returns (PRIMARY pitch
    + by-window + SECONDARY q0/rax + meta), so the full structure is preserved.
    """

    name: str
    source: str  # "live" | "from_reference"
    pitch_rmse: float
    pitch_crps: float
    pitch_nll: float
    pitch_cov90: float
    n_shots: int
    scored: dict = field(default_factory=dict)
    pitch_rmse_ci: tuple[float, float] | None = None

    def summary(self) -> dict[str, Any]:
        """Flat, JSON-friendly headline (the comparable row)."""
        return {
            "name": self.name,
            "source": self.source,
            "pitch_rmse": self.pitch_rmse,
            "pitch_rmse_ci": list(self.pitch_rmse_ci) if self.pitch_rmse_ci else None,
            "pitch_crps": self.pitch_crps,
            "pitch_nll": self.pitch_nll,
            "pitch_cov90": self.pitch_cov90,
            "n_shots": self.n_shots,
        }


@dataclass
class Bar:
    """The raw-held-out prediction bar: persistence + EnKF, one ruler.

    ``legs`` maps predictor name -> :class:`BarLeg`.  :attr:`target` is the
    pitch RMSE the world model's raw-held-out prediction must beat — the EnKF
    leg's RMSE (the credible classical comparator), with persistence as the
    sanity floor below it.
    """

    legs: dict[str, BarLeg] = field(default_factory=dict)
    coverage_gate: tuple[float, float] = REFERENCE["coverage_gate"]
    physics_frontier_pitch_rmse: float = REFERENCE["physics_frontier_pitch_rmse"]
    near_axis_rad_floor: float = REFERENCE["near_axis_rad_floor"]

    @property
    def target(self) -> float | None:
        """The pitch RMSE the model must beat — the EnKF leg if present."""
        leg = self.legs.get("enkf")
        return leg.pitch_rmse if leg is not None else None

    @property
    def floor(self) -> float | None:
        """The sanity floor — persistence pitch RMSE if present."""
        leg = self.legs.get("persistence")
        return leg.pitch_rmse if leg is not None else None

    def beats(self, predictor_pitch_rmse: float, *, against: str = "enkf") -> bool:
        """Does a candidate pitch RMSE beat (i.e. fall below) a leg's RMSE?"""
        leg = self.legs.get(against)
        if leg is None or not _finite(leg.pitch_rmse):
            return False
        return float(predictor_pitch_rmse) < leg.pitch_rmse

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "raw-held-out-prediction-bar-v0",
            "legs": {name: leg.summary() for name, leg in self.legs.items()},
            "target_pitch_rmse": self.target,
            "floor_pitch_rmse": self.floor,
            "coverage_gate": list(self.coverage_gate),
            "physics_frontier_pitch_rmse": self.physics_frontier_pitch_rmse,
            "near_axis_rad_floor": self.near_axis_rad_floor,
            "reference": REFERENCE,
        }


def _finite(x: Any) -> bool:
    try:
        import math  # noqa: PLC0415

        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Leg builders
# ---------------------------------------------------------------------------


def _leg_from_scored(name: str, source: str, scored: dict) -> BarLeg:
    """Hoist a harness ``score(...)`` result into a flat :class:`BarLeg`."""
    pp = scored.get("primary", {}).get("pitch", {})
    return BarLeg(
        name=name,
        source=source,
        pitch_rmse=float(pp.get("rmse", float("nan"))),
        pitch_crps=float(pp.get("crps", float("nan"))),
        pitch_nll=float(pp.get("nll", float("nan"))),
        pitch_cov90=float(pp.get("cov90", float("nan"))),
        n_shots=int(pp.get("n_shots", 0)),
        scored=scored,
    )


def persistence_bar(
    manifest: dict,
    truth: mse_eval.MseTruth | None = None,
) -> BarLeg:
    """Score the persistence reference on a held-out manifest (CHEAP, live).

    Persistence freezes the pitch profile at the first beam-on slice — no
    solver, no GPU.  It is the sanity floor the EnKF (and the world model) must
    clear.  ``truth`` defaults to the level-1 corpus loader; tests pass a
    synthetic stand-in.
    """
    if truth is None:
        # 2-arg contract: harness loads truth on demand from level-1.
        truth = mse_eval.MseTruth(level1_dir=_default_level1_dir())
    predictor = mse_eval.PersistencePredictor()
    preds = predictor.predict(manifest, truth)
    scored = mse_eval.score(preds, manifest, truth)
    return _leg_from_scored("persistence", "live", scored)


def _default_level1_dir() -> Path:
    from imas_ambix.data.paths import LEVEL1_DIR  # noqa: PLC0415

    return LEVEL1_DIR


def enkf_leg_from_reference(
    artifact_path: Path | None = None,
    *,
    arm: str = "analysis",
) -> BarLeg:
    """Read the EnKF leg from the locked reference metrics artifact.

    Used when the heavy live EnKF run cannot complete in-session.  The artifact
    stores a bootstrap-CI block per metric (``{"mean", "ci_lo", "ci_hi", "n"}``)
    for the full 112-shot D1 held-out set; this hoists the analysis arm into a
    :class:`BarLeg` flagged ``"from_reference"``.  ``arm`` selects ``analysis``
    (the scored, magnetics-assimilated arm) or ``forecast`` (the non-vacuity
    control).
    """
    path = artifact_path or _REFERENCE_ARTIFACT
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key = (
        "metrics_analysis_arm"
        if arm == "analysis"
        else "metrics_forecast_arm_NONVACUITY_CONTROL"
    )
    block = payload.get(key, {})

    def _mean(metric: str) -> float:
        cell = block.get(metric)
        if isinstance(cell, dict):
            return float(cell.get("mean", float("nan")))
        return float(cell) if cell is not None else float("nan")

    def _ci(metric: str) -> tuple[float, float] | None:
        cell = block.get(metric)
        if isinstance(cell, dict) and "ci_lo" in cell and "ci_hi" in cell:
            return (float(cell["ci_lo"]), float(cell["ci_hi"]))
        return None

    n_cell = block.get("pitch_rmse")
    n_shots = (
        int(n_cell.get("n", block.get("n_shots", 0)))
        if isinstance(n_cell, dict)
        else int(block.get("n_shots", 0))
    )
    return BarLeg(
        name="enkf",
        source="from_reference",
        pitch_rmse=_mean("pitch_rmse"),
        pitch_crps=_mean("pitch_crps"),
        pitch_nll=_mean("pitch_nll"),
        pitch_cov90=_mean("pitch_cov90"),
        n_shots=n_shots,
        scored={"reference_artifact": str(path), "arm": arm, "block": block},
        pitch_rmse_ci=_ci("pitch_rmse"),
    )


def enkf_bar_live(
    manifest: dict,
    shot_ids,
    cfg: EnKFConfig | None = None,
    truth: mse_eval.MseTruth | None = None,
    *,
    arm: str = "analysis",
) -> BarLeg:
    """Score the EnKF reference LIVE on the held-out manifest (HEAVY).

    Runs the TORAX-referenced ensemble smoother for ``shot_ids`` and scores the
    resulting predictions on the SAME harness as persistence.  HEAVY — per the
    repo AGENTS.md the EnKF (CPU/assimilation, TORAX-referenced) belongs on the
    ``sun_debug`` SLURM partition with ``TMPDIR=/tmp``, not the GPU reservation.
    Most callers use :func:`enkf_leg_from_reference` to fill this leg from the
    locked artifact and flag the live re-measure as a followup.
    """
    from imas_ambix.statespace import enkf_baseline as enkf  # noqa: PLC0415

    cfg = cfg or enkf.EnKFConfig()
    truth = truth or mse_eval.MseTruth(level1_dir=_default_level1_dir())
    grid = {
        int(sid): {
            "t": _np().asarray(
                manifest["shots"][str(int(sid))]["beam_on_slice_times"],
                dtype=float,
            ),
            "rpos": _np().asarray(
                manifest["shots"][str(int(sid))]["active_channel_rpos"],
                dtype=float,
            ),
        }
        for sid in shot_ids
        if str(int(sid)) in manifest["shots"]
    }
    preds = enkf.predict_shots(shot_ids, cfg, arm=arm, manifest_grid=grid)
    scored = mse_eval.score(preds, manifest, truth)
    return _leg_from_scored("enkf", "live", scored)


def _np():
    import numpy as np  # noqa: PLC0415

    return np


# ---------------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------------


def prediction_bar(
    manifest: dict,
    *,
    truth: mse_eval.MseTruth | None = None,
    enkf_shot_ids=None,
    enkf_cfg: EnKFConfig | None = None,
    run_enkf_live: bool = False,
    reference_artifact: Path | None = None,
) -> Bar:
    """Assemble the raw-held-out {persistence, EnKF} prediction bar.

    Persistence is ALWAYS scored live (cheap).  The EnKF leg is filled from the
    locked reference artifact (``run_enkf_live=False``, the default) or scored
    live (``run_enkf_live=True`` — heavy; needs TORAX + the sun_debug partition).

    Parameters
    ----------
    manifest:
        The held-out split manifest (the D1 locked split, or a synthetic
        stand-in for tests).
    truth:
        Truth provider; defaults to the level-1 corpus loader.
    enkf_shot_ids:
        Shots to run the live EnKF on (only used when ``run_enkf_live``).
        Defaults to the manifest's held-out shots.
    run_enkf_live:
        Run the heavy live EnKF instead of reading the reference artifact.
    reference_artifact:
        Override path to the locked EnKF metrics artifact.
    """
    legs: dict[str, BarLeg] = {}
    legs["persistence"] = persistence_bar(manifest, truth)

    if run_enkf_live:
        if enkf_shot_ids is None:
            enkf_shot_ids = [
                int(sid)
                for sid, e in manifest.get("shots", {}).items()
                if e.get("partition") == "held_out"
            ]
        legs["enkf"] = enkf_bar_live(manifest, enkf_shot_ids, enkf_cfg, truth)
    else:
        legs["enkf"] = enkf_leg_from_reference(reference_artifact)

    return Bar(legs=legs)


# ---------------------------------------------------------------------------
# Split / manifest helpers (thin wrappers around the locked D1 split)
# ---------------------------------------------------------------------------


def load_locked_manifest(path: Path | None = None) -> dict:
    """Load the locked D1 held-out split manifest (leakage-audited eval split)."""
    if path is None:
        from imas_ambix.data.paths import MANIFEST_DIR  # noqa: PLC0415

        path = MANIFEST_DIR / "mse_heldout_split_v0.json"
    return mse_eval.load_manifest(Path(path))


def held_out_shot_ids(manifest: dict) -> list[int]:
    """The held-out shot ids of a manifest (the scored set)."""
    return sorted(
        int(sid)
        for sid, e in manifest.get("shots", {}).items()
        if e.get("partition") == "held_out"
    )


# ---------------------------------------------------------------------------
# CLI — report the live persistence bar + the EnKF bar from the reference
# ---------------------------------------------------------------------------


def _main() -> int:
    """Report the persistence (live) + EnKF (reference) raw-held-out bar."""
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="held-out split manifest (default: locked D1 mse_heldout_split_v0)",
    )
    ap.add_argument(
        "--run-enkf-live",
        action="store_true",
        help="run the heavy live EnKF instead of reading the reference artifact",
    )
    args = ap.parse_args()

    manifest = (
        mse_eval.load_manifest(args.manifest)
        if args.manifest
        else load_locked_manifest()
    )
    bar = prediction_bar(manifest, run_enkf_live=args.run_enkf_live)
    print(json.dumps(bar.to_dict(), indent=2, default=float))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(_main())


__all__ = [
    "Bar",
    "BarLeg",
    "REFERENCE",
    "persistence_bar",
    "enkf_bar_live",
    "enkf_leg_from_reference",
    "prediction_bar",
    "load_locked_manifest",
    "held_out_shot_ids",
]
