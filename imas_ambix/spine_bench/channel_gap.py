"""Decompose the frozen gate's sensor misfit onto the channels that carry it.

``magnetics_residual_whitened_rms`` is one number per stamp: the median over
slices of a whitened rms over every mapped channel.  That is the right shape for
a gate and the wrong shape for deciding what to fix, because a pooled rms cannot
say whether the misfit is spread thinly over seventy-seven channels or is two
probes standing well outside the array.

This runs the same arm the gate scores -- the same frozen shots, the same spine
config, the same two-phase solve, the same converged equilibria -- and keeps the
residual per channel instead of collapsing it.  Each channel is then reported in
the units it was measured in, beside the noise floor measured for it on quiescent
windows of plasma-free shots, so a channel's remaining gap can be read as a
multiple of what the instrument itself scatters by.  A channel already at its
floor is finished; one at ten times its floor is a description error with room to
find.

Run it as::

    AMBIX_MACHINE_ARTIFACT_CACHE=... AMBIX_MACHINE_ARTIFACT_DIGEST=sha256:... \\
        python -m imas_ambix.spine_bench.channel_gap --source artifact

Only the primary ``greens-matvec`` arm is decomposed: it is the arm the residual
tolerances gate, and the grid-Δ* arm agrees with it to the fourth decimal, so a
second decomposition would restate the same channels.

The flux loops carry webers and the b-probes tesla, so the physical column is
per-family and only the b-probes have a measured field floor to be compared with.
A loop is reported with its whitened residual and no ratio, which is honest about
there being no flux-unit floor measurement rather than silently mixing units.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from imas_ambix.latent.gs_solve import SUBSTRATE_GREENS
from imas_ambix.spine_bench.runner import CONFINED_AXIS_R_MAX
from imas_ambix.spine_bench.schema import SCHEMA_VERSION
from imas_ambix.spine_bench.shots import FROZEN_SHOTSET, resolve_shotset_version

logger = logging.getLogger(__name__)

#: Where nova banked the per-channel floors measured on quiescent windows.
NOISE_ENVELOPE_PATH = Path.home() / ".cache/nova-mast/mast_sensor_noise.json"


def _noise_floors(path: Path = NOISE_ENVELOPE_PATH) -> dict[str, float]:
    """Return each b-probe's measured field floor [T], empty when unavailable.

    Read as data rather than recomputed: the envelope rests on sixty quiescent
    shots and measuring it again here would be a different measurement wearing
    the same name.
    """
    if not path.exists():
        logger.warning("no noise envelope at %s -- ratios will be omitted", path)
        return {}
    record = json.loads(path.read_text())
    return {
        row["channel"]: float(row["scatter"]) for row in record["envelope"]["channels"]
    }


def summarise_channels(
    whitened: dict[str, list[float]],
    physical: dict[str, list[float]],
    floors: dict[str, float],
) -> list[dict[str, Any]]:
    """Rank the channels by how much of the pooled misfit each one carries.

    The share is of the MEAN SQUARE, not of the rms, because that is the quantity
    the pooled residual is an average of and it is therefore the only split that
    adds to one.  Ranking by rms instead would let a channel measured on few
    slices outrank one carrying more of the total.

    ``gap_over_floor`` divides a channel's residual by what the instrument itself
    scatters by, so it answers the question the ladder asks -- is there anything
    left to find on this channel -- in a form that does not depend on the
    whitening convention.  It is omitted rather than faked where no floor was
    measured in that channel's units.
    """
    total_square = sum(
        float(np.sum(np.asarray(rows, dtype=float) ** 2)) for rows in whitened.values()
    )
    rows: list[dict[str, Any]] = []
    for channel in sorted(whitened):
        w = np.asarray(whitened[channel], dtype=float)
        q = np.asarray(physical[channel], dtype=float)
        rms_physical = float(np.sqrt(np.mean(q**2)))
        floor = floors.get(channel)
        rows.append(
            {
                "channel": channel,
                "n_slices": int(w.size),
                "rms_whitened": float(np.sqrt(np.mean(w**2))),
                "median_whitened": float(np.median(np.abs(w))),
                "rms_physical": rms_physical,
                "unit": "Wb" if channel.startswith("fl") else "T",
                "noise_floor": None if floor is None else float(floor),
                "gap_over_floor": None if not floor else float(rms_physical / floor),
                "share_of_mean_square": (
                    float(np.sum(w**2) / total_square) if total_square else 0.0
                ),
            }
        )
    rows.sort(key=lambda row: -row["share_of_mean_square"])
    return rows


def channel_residuals(source, *, max_slices: int = 6, sigma: float = 0.02) -> dict:
    """Return the per-channel residual of every scored slice of the frozen set.

    Mirrors :func:`imas_ambix.spine_bench.runner.run_stamp`'s primary arm exactly
    -- disc seed, basin solve, profile solve, then the vacuum prediction plus the
    cell-to-sensor matvec of the converged plasma currents -- so the numbers here
    add up to the residual that stamp reports rather than to a similar quantity.
    """
    from imas_ambix.latent.boundary_disc import disc_read
    from imas_ambix.latent.gs_solve import EquilibriumGrid
    from scripts.greens_filament_gate_eval import _fit_slice
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import factory_shot_payloads, frozen_spine_config

    spine, spine_sha = frozen_spine_config()
    iso = spine["interior_solve"]
    spine_kw = dict(
        n_p=int(iso["n_p"]),
        n_f=int(iso["n_f"]),
        nonneg=iso["profile_kind"] == "monomial-nonneg",
        smoothness=float(iso["smoothness"]),
        boundary_read=iso["boundary_read_scoring"],
        sigma=sigma,
    )

    whitened: dict[str, list[float]] = {}
    physical: dict[str, list[float]] = {}
    slice_rms: list[float] = []
    warrants_by_shot: dict[int, tuple] = {}
    n_slices = 0
    for bench_shot in FROZEN_SHOTSET:
        shot = int(bench_shot.shot_id)
        table = source.table_for(shot)
        if table is None:
            continue
        payload = factory_shot_payloads(
            shot, nr=65, nz=97, max_slices=max_slices, min_ip_ka=200.0, table=table
        )
        if payload is None:
            continue
        warrants_by_shot[shot] = tuple(payload.get("scale_corrections", ()))
        grid_gs, tbl, basis = payload["grid"], payload["table"], payload["basis"]
        grid = EquilibriumGrid.from_table(tbl, nr=65, nz=97)
        g_sens, channels = grid.sensor_greens(tbl)
        cell_area = grid.dr * grid.dz
        for k in np.argsort([p.time_s for p in payload["payloads"]]):
            p = payload["payloads"][int(k)]
            inv = disc_read(p, grid_gs, tbl, basis)
            if inv is None or inv.ring is None:
                continue
            centroid = (float(inv.centroid_r), float(inv.centroid_z))
            seed = _disc_seed_flat(grid_gs, inv)
            basin = _fit_slice(
                grid,
                tbl,
                basis,
                p,
                substrate=SUBSTRATE_GREENS,
                warm=seed,
                centroid=centroid,
                n_p=1,
                n_f=1,
                nonneg=False,
                smoothness=spine_kw["smoothness"],
                boundary_read=spine_kw["boundary_read"],
                sigma=sigma,
                topology_read="hard",
            )
            if (
                basin.scored
                and basin.jphi_flat is not None
                and np.isfinite(basin.target[0])
                and basin.target[0] <= CONFINED_AXIS_R_MAX
            ):
                seed = basin.jphi_flat
            fit = _fit_slice(
                grid,
                tbl,
                basis,
                p,
                substrate=SUBSTRATE_GREENS,
                warm=seed,
                centroid=centroid,
                topology_read="hard",
                **spine_kw,
            )
            if not (fit.scored and fit.jphi_flat is not None):
                continue
            i_cell = np.asarray(fit.jphi_flat, dtype=np.float64)[grid.cells] * cell_area
            pred = np.asarray(p.vacuum, dtype=np.float64) + g_sens @ i_cell
            meas = np.asarray(p.measured, dtype=np.float64)
            scale = np.asarray(p.scale, dtype=np.float64)
            ok = (
                np.asarray(p.mask, dtype=bool)
                & np.isfinite(meas)
                & np.isfinite(pred)
                & (scale > 0.0)
            )
            if not ok.any():
                continue
            n_slices += 1
            gap = pred - meas
            slice_rms.append(float(np.sqrt(np.mean((gap[ok] / scale[ok]) ** 2))))
            for column in np.flatnonzero(ok):
                channel = channels[int(column)]
                whitened.setdefault(channel, []).append(
                    float(gap[column] / scale[column])
                )
                physical.setdefault(channel, []).append(float(gap[column]))

    floors = _noise_floors()
    rows = summarise_channels(whitened, physical, floors)
    return {
        "schema_version": SCHEMA_VERSION,
        "shotset_version": resolve_shotset_version(list(FROZEN_SHOTSET)),
        "created_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "git_commit": _commit(),
        "engine_config_sha": spine_sha,
        "arm": SUBSTRATE_GREENS,
        "geometry_source": source.label,
        "geometry_revision": source.revision,
        "geometry_provenance": source.provenance(),
        "measurement_read": _read_record(warrants_by_shot),
        "n_slices_scored": n_slices,
        # the metric the stamp reports, recomputed here as a cross-check that
        # this decomposition ran on the same equilibria
        "median_slice_rms": float(np.median(slice_rms)) if slice_rms else float("nan"),
        "pooled_noise_floor_t": _pooled_floor(floors),
        "channels": rows,
    }


def _read_record(warrants_by_shot: dict[int, tuple]) -> dict[str, Any]:
    from imas_ambix.spine_bench.runner import _measurement_read

    return _measurement_read(warrants_by_shot)


def _pooled_floor(floors: dict[str, float]) -> float:
    if not floors:
        return float("nan")
    values = np.asarray(sorted(floors.values()), dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return ""


def _source(name: str):
    if name == "artifact":
        from imas_ambix.spine_bench.machine_artifact_arm import source_from_environment

        source = source_from_environment()
        source.build()
        return source
    from imas_ambix.spine_bench.runner import CampaignGeometrySource

    return CampaignGeometrySource()


def main(argv: list[str] | None = None) -> int:
    """Write one arm's per-channel decomposition beside the banked stamps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("artifact", "campaign"), default="artifact"
    )
    parser.add_argument("--max-slices", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "results")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    record = channel_residuals(
        _source(args.source), max_slices=args.max_slices, sigma=args.sigma
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (
        f"channel-gap-{record['shotset_version']}-{record['git_commit']}"
        f"-{record['hostname']}-{args.source}.json"
    )
    path.write_text(json.dumps(record, indent=1, sort_keys=False) + "\n")
    top = record["channels"][:6]
    logger.info("median slice rms: %.7f", record["median_slice_rms"])
    for row in top:
        logger.info(
            "%-8s share %5.1f%%  rms %.4g %s  floor %s  ratio %s",
            row["channel"],
            100.0 * row["share_of_mean_square"],
            row["rms_physical"],
            row["unit"],
            "n/a" if row["noise_floor"] is None else f"{row['noise_floor']:.3g}",
            "n/a" if row["gap_over_floor"] is None else f"{row['gap_over_floor']:.1f}x",
        )
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
