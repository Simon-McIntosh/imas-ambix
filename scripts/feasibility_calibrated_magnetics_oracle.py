#!/usr/bin/env python
"""CLI: does ABSOLUTE / CALIBRATED magnetics resolve plasma geometry?

A thin entrypoint over :mod:`imas_ambix.eval.magnetics_oracle`.  It reads the
PRE-quantised FLOAT magnetics channels straight from the staged stores, fits a
SINGLE corpus-level mean / std per channel on the TRAIN shots (preserving the
ABSOLUTE inter-shot field magnitude — the whole point of the experiment),
resamples each channel onto the camera-window grid, and trains a thin temporal
probe with a Gaussian head over the equilibrium geometry target.

The reported skill (probe geometry RMSE vs a mean-predictor baseline, per
component and headline axis+X-point mean) is the MEASURED BAR a learned
world-model's stage-2 Grad-Shafranov readout must match.

EVALUATOR-ONLY (binding firewall)
---------------------------------
A third-party evaluator.  The probe input is raw measured magnetics; the LABEL
is the L2 equilibrium.  Nothing here is, or is importable by, the world-model
training path — it only consumes data and produces evaluator metrics.  No WM
checkpoint is loaded.  Equilibrium is an evaluator label only.

Outputs (JSON + a pred-vs-true axis / X-point scatter) under the chosen output
root and ``docs/figures/joint-multimodal-plasma-wm/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from imas_ambix.eval.magnetics_oracle import (
    RAW_CHANNEL_KEYS,
    OracleConfig,
    geometry_scatter,
    run_oracle,
)

logger = logging.getLogger("calib_mag_oracle")

DEFAULT_OUT_ROOT = Path(
    "/work/projects/imas_gpu/worldmodel/calibrated_magnetics_oracle"
)
DEFAULT_FIG_DIR = Path("docs/figures/joint-multimodal-plasma-wm")


def run(args) -> int:
    config = OracleConfig(
        camera=args.camera,
        n_frames=args.n_frames,
        target_horizon_s=args.target_horizon_s,
        n_signal_steps=args.n_signal_steps,
        n_train_shots=args.n_train_shots,
        n_test_shots=args.n_test_shots,
        held_out_fraction=args.held_out_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        ratio_threshold=args.ratio_threshold,
        seed=args.seed,
    )
    token_root = Path(args.token_root) if args.token_root else None
    level2_root = Path(args.level2_root) if args.level2_root else None

    try:
        result = run_oracle(config, token_root=token_root, level2_root=level2_root)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    verd = result.verdict
    logger.info(
        "=== CALIBRATED-MAGNETICS VERDICT: %s ===",
        "FEASIBLE" if verd.feasible else "INFEASIBLE",
    )
    for row in verd.components:
        logger.info(
            "  %-10s probe=%s  baseline=%s  skill=%s  beats=%s",
            row["component"],
            "n/a"
            if row["rmse_probe_m"] is None
            else f"{row['rmse_probe_m'] * 100:.1f}cm",
            "n/a"
            if row["rmse_baseline_m"] is None
            else f"{row['rmse_baseline_m'] * 100:.1f}cm",
            "n/a" if row["skill"] is None else f"{row['skill']:+.2f}",
            row["beats_baseline"],
        )
    logger.info(
        "headline axis+X-point skill=%s  (axis=%s  X-point=%s)",
        "n/a" if verd.headline_skill is None else f"{verd.headline_skill:+.3f}",
        "n/a" if verd.axis_skill is None else f"{verd.axis_skill:+.3f}",
        "n/a" if verd.xpt_skill is None else f"{verd.xpt_skill:+.3f}",
    )

    report = {
        "task": (
            "calibrated-magnetics feasibility oracle (RAW physical-unit "
            "magnetics -> plasma geometry; corpus-level standardisation, "
            "absolute scale preserved)"
        ),
        "evaluator_only": True,
        "input": "raw_calibrated_magnetics_floats",
        "standardisation": "single corpus-level mean/std per channel (TRAIN-fit)",
        "camera": config.camera,
        "n_frames": config.n_frames,
        "n_signal_steps": config.n_signal_steps,
        "epochs": config.epochs,
        "target_names": list(result.target_names),
        "target_units": "m",
        "coverage": {
            "train_examples": result.train_examples,
            "test_examples": result.test_examples,
            "train_shots": result.train_shots,
            "test_shots": result.test_shots,
            "forced_test_present": result.forced_test_present,
            "raw_channels": result.n_channels,
            "raw_channel_keys": list(RAW_CHANNEL_KEYS),
            "n_signal_steps": config.n_signal_steps,
            "target_horizon_s": config.target_horizon_s,
        },
        "probe_params_M": result.probe_params_millions,
        "verdict": verd.to_dict(),
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "oracle_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", json_path)

    title = (
        "calibrated-magnetics feasibility oracle — RAW physical-unit magnetics "
        "-> plasma geometry"
    )
    fig_local = out_root / "fig-calib-mag-oracle-geometry-scatter.png"
    geometry_scatter(
        result.pred,
        result.y_test,
        result.mask_test,
        result.target_names,
        fig_local,
        title=title,
    )
    fig_docs = Path(args.fig_dir) / "fig-calib-mag-oracle-geometry-scatter.png"
    try:
        geometry_scatter(
            result.pred,
            result.y_test,
            result.mask_test,
            result.target_names,
            fig_docs,
            title=title,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write docs figure %s: %s", fig_docs, exc)

    logger.info(
        "=== TOP-LEVEL (calibrated-magnetics) FEASIBILITY: %s ===",
        "FEASIBLE" if verd.feasible else "INFEASIBLE",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    d = OracleConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default=d.camera)
    p.add_argument(
        "--n-frames", type=int, default=d.n_frames, help="camera frames per window"
    )
    p.add_argument(
        "--target-horizon-s",
        type=float,
        default=d.target_horizon_s,
        help="physical time span a window covers (s)",
    )
    p.add_argument(
        "--n-signal-steps",
        type=int,
        default=d.n_signal_steps,
        help="raw-magnetics temporal positions across the window span",
    )
    p.add_argument("--n-train-shots", type=int, default=d.n_train_shots)
    p.add_argument("--n-test-shots", type=int, default=d.n_test_shots)
    p.add_argument("--held-out-fraction", type=float, default=d.held_out_fraction)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--d-model", type=int, default=d.d_model)
    p.add_argument("--n-layers", type=int, default=d.n_layers)
    p.add_argument("--n-heads", type=int, default=d.n_heads)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=d.ratio_threshold,
        help="probe must beat baseline by this factor on axis+X-point",
    )
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--token-root", default=None, help="override token root")
    p.add_argument("--level2-root", default=None, help="override L2 equilibrium root")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
