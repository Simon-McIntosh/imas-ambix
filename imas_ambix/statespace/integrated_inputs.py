"""Integrated (wider) input loaders for the GS-grounded engine — T10.

Why this module exists
----------------------
T6 showed the magnetics-only latent cannot ground the *internal* current
profile (near-vacuum c_plasma ratio 0.98 FAIL): external magnetics fix the
boundary + total Ip + low moments, NOT the internal p′/FF′ terms.  The user's
redirect (decision ``integrated-input-set = internal-core``) widens the
engine's input set with the two internal-profile diagnostics the GS equation
needs and magnetics lack:

    * MSE (ams)      → internal current / q  (the FF′ term)
    * Thomson (TS)   → kinetic pressure p(ψ) (the p′ term)

A LOAD-BEARING CORPUS FINDING (T10, verified across 25/25 co-available shots)
-----------------------------------------------------------------------------
The level-1 ``ams`` group does NOT carry a time-resolved MSE pitch-angle /
internal-current signal.  Its only payload, ``acoeff``, has the array attr
``shape = [1, 6, N]`` with ``time_index = 0`` — i.e. a SIZE-1 time dimension:
it is the static per-shot **a-coefficient geometry table** (6 MSE viewing
channels × N a-coefficients linking polarisation angle γ to B), NOT γ(t).  No
γ/pitch/polarisation time series exists in ``ams`` or any sibling group (grep
across all shot groups returns only this static table + an unrelated Thomson
geometry ``angle`` + the saddle-coil ``asm`` mode detector).  The actual
internal-current constraint MSE would provide is therefore **unrealizable from
this corpus** — T9's co-availability scoping counted ``ams`` *present* (true)
but the content is geometry, not a current measurement.

Consequence (surfaced honestly, not silently absorbed): the locked
``internal-core`` set is **half-unrealizable** from level-1.  This module wires
the half that IS realizable — Thomson pressure ``pe(t, R)`` — as a compact,
gridding-invariant per-measurement-time feature vector, held at the native
~5 ms Thomson cadence (forward-filled onto the 1 kHz engine grid between
measurements, NEVER interpolated as if dense), weighted per shot by profile
completeness (full-profile ayc/atm vs edge-only aye).  This is the honest
"pressure-only internal grounding" variant the T10 prompt anticipates for the
branch "report which internal constraint is still missing".

Thomson feature design (design-for-invariance honoured)
-------------------------------------------------------
The three TS systems differ in radial gridding (atm 35, ayc 131, aye 16) and
coverage (ayc/atm full-profile R∈[0.24,1.5] m, aye EDGE-ONLY R∈[1.27,1.47] m),
so we reduce each system's ``pe(t, R)`` to ONE fixed feature vector per
measurement time, gridding-invariant by construction:

    * pe interpolated onto a FIXED vessel-major-radius grid (8 nodes over
      R∈[0.2,1.5] m — vessel geometry only, NEVER an EFIT ψ map);
    * 5 robust profile scalars: peak pressure, pressure-weighted centroid
      radius, trapezoidal integral, edge/core ratio, radial extent of valid
      points (the completeness signal);
    * a per-time validity flag (1 if the measurement carried any finite pe).

The forward-fill carries the LAST measured profile between Thomson shots, with
the validity flag decaying so the engine can distinguish "fresh pressure
measurement" from "stale held value" — this is the sparse-likelihood weighting
at the *input* level (the GS prior weighting the pressure LIKELIHOOD at
measurement times only is the documented next increment — see the T10 verdict;
it requires a head z→p(ψ) + a flux-surface map and is out of scope for the
focused single-config run).

Scope: import-only from ``statespace.baseline`` (the mag+ane loader + schema)
and ``statespace.splits`` / ``gs.integrated_inventory`` (the locked split
machinery).  CPU; pure numpy.  Keeps the ungrounded + v2 mag-ane paths intact
(this module is ADDITIVE — the engine only uses it when an integrated schema is
passed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thomson pressure feature schema (gridding-invariant, fixed vessel coords)
# ---------------------------------------------------------------------------

# Fixed vessel-major-radius grid for the interpolated pressure profile.  Vessel
# geometry only (MAST R∈~[0.2,1.5] m) — NOT an EFIT flux-surface map (honours
# the never-efm-output lock + design-for-invariance fixed-geometry rule).
TS_R_GRID: np.ndarray = np.linspace(0.20, 1.50, 8)  # (8,) major radius nodes (m)

# Per-measurement Thomson feature layout (built by ``thomson_features_at_time``).
TS_PROFILE_CHANNELS: list[str] = [f"pe_R{r:.2f}" for r in TS_R_GRID]
TS_SCALAR_CHANNELS: list[str] = [
    "pe_peak",  # max valid pe in the profile
    "pe_centroid_R",  # pressure-weighted mean radius (profile centre)
    "pe_integral",  # trapezoidal ∫ pe dR over valid points
    "pe_edge_core_ratio",  # edge / core pressure (peakedness proxy)
    "pe_radial_extent",  # R_max − R_min of valid points (completeness signal)
]
# A per-time freshness flag: 1.0 at a fresh Thomson measurement, decaying toward
# 0 as the held value goes stale (so the engine sees sparse-cadence structure).
TS_FRESHNESS_CHANNEL: str = "ts_fresh"

THOMSON_FEATURE_CHANNELS: list[str] = [
    *TS_PROFILE_CHANNELS,
    *TS_SCALAR_CHANNELS,
    TS_FRESHNESS_CHANNEL,
]
N_THOMSON_FEATURES: int = len(THOMSON_FEATURE_CHANNELS)  # 8 + 5 + 1 = 14

# Robust pressure clip (level-1 atm carries occasional huge/negative outliers;
# the median valid MAST e-pressure proxy is O(1e2–1e3)).  Clip BEFORE features.
_PE_CLIP_MIN = 0.0
_PE_CLIP_MAX = 1.0e5
# Freshness decay time constant (s): a held profile is "stale" after a few ms.
_TS_FRESH_TAU_S = 6.0e-3
# A profile counts as "full" (completeness 1.0) when its valid radial extent
# covers at least this fraction of the vessel grid span; edge-only (aye)
# profiles cover much less and are down-weighted accordingly.
_FULL_PROFILE_EXTENT_FRAC = 0.5
_TS_R_SPAN = float(TS_R_GRID[-1] - TS_R_GRID[0])


# ---------------------------------------------------------------------------
# Per-measurement Thomson profile → fixed feature vector
# ---------------------------------------------------------------------------


def thomson_features_at_time(pe_row: np.ndarray, r_row: np.ndarray) -> np.ndarray:
    """Reduce one Thomson ``pe(R)`` profile to the fixed (14,) feature vector.

    Parameters
    ----------
    pe_row : (R,) electron-pressure values at this measurement time.
    r_row  : (R,) major-radius (m) of each pressure point.

    Returns a (N_THOMSON_FEATURES,) vector.  All-NaN / empty profiles return a
    zero vector (with freshness 0) — the caller decides forward-fill behaviour.
    """
    feat = np.zeros(N_THOMSON_FEATURES, dtype=np.float64)
    pe = np.asarray(pe_row, dtype=np.float64)
    r = np.asarray(r_row, dtype=np.float64)
    m = np.isfinite(pe) & np.isfinite(r)
    if m.sum() < 2:
        return feat  # no usable profile → zero features, freshness 0
    pe_v = np.clip(pe[m], _PE_CLIP_MIN, _PE_CLIP_MAX)
    r_v = r[m]
    order = np.argsort(r_v)
    r_v, pe_v = r_v[order], pe_v[order]

    # (a) profile on the fixed vessel-R grid (linear interp; flat-extrapolate)
    prof = np.interp(TS_R_GRID, r_v, pe_v, left=pe_v[0], right=pe_v[-1])
    n_prof = len(TS_R_GRID)
    feat[:n_prof] = prof

    # (b) robust scalars
    pe_peak = float(np.max(pe_v))
    pe_sum = float(np.sum(pe_v))
    centroid = (
        float(np.sum(r_v * pe_v) / pe_sum) if pe_sum > 1e-12 else float(np.mean(r_v))
    )
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy≥2 renamed it
    integral = float(_trapz(pe_v, r_v))
    # edge/core: mean pe in outer third vs inner third of the VALID radial span
    r_lo, r_hi = r_v[0], r_v[-1]
    third = (r_hi - r_lo) / 3.0
    core = pe_v[r_v <= r_lo + third]
    edge = pe_v[r_v >= r_hi - third]
    core_m = float(np.mean(core)) if core.size else pe_peak
    edge_m = float(np.mean(edge)) if edge.size else 0.0
    edge_core = float(edge_m / core_m) if abs(core_m) > 1e-12 else 0.0
    extent = float(r_hi - r_lo)

    feat[n_prof : n_prof + 5] = [pe_peak, centroid, integral, edge_core, extent]
    feat[-1] = 1.0  # freshness: this is a fresh measurement
    return feat


def _profile_completeness(extent: float) -> float:
    """Per-profile completeness weight from valid radial extent (full vs edge).

    Full-profile (ayc/atm) covers most of the vessel grid → ~1.0; edge-only
    (aye) covers a small outer band → small.  Clipped to [0, 1].
    """
    return float(np.clip(extent / (_FULL_PROFILE_EXTENT_FRAC * _TS_R_SPAN), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-shot Thomson feature stream on the engine grid (held at native cadence)
# ---------------------------------------------------------------------------


@dataclass
class ThomsonStream:
    """Thomson features aligned to a shot's engine grid + provenance."""

    features: np.ndarray  # (T_grid, N_THOMSON_FEATURES) held at native cadence
    completeness: float  # per-shot profile-completeness weight in [0, 1]
    system: str  # which TS system was used (ayc|atm|aye) or "none"
    n_measurements: int  # number of fresh Thomson measurement times found


# preference order: full-profile combined (ayc) > core (atm) > edge-only (aye)
_TS_SYSTEM_PRIORITY: tuple[tuple[str, str], ...] = (
    ("ayc", "radius"),
    ("atm", "radius"),
    ("aye", "r"),
)


def load_thomson_stream(
    store,
    grid_times: np.ndarray,
) -> ThomsonStream:
    """Build the (T_grid, 14) Thomson feature stream for one shot.

    Picks the highest-priority TS system present (ayc→atm→aye), extracts the
    fixed feature vector at each VALID measurement time, then forward-fills onto
    ``grid_times`` (carrying the last measured profile between Thomson shots,
    with the freshness flag decaying).  Returns a zero-feature stream (system
    "none", completeness 0) if no TS system has a usable profile.

    ``store`` is an open Zarr group for the shot; ``grid_times`` is the engine's
    1 kHz common grid (seconds).
    """
    T = len(grid_times)
    empty = ThomsonStream(
        features=np.zeros((T, N_THOMSON_FEATURES), dtype=np.float64),
        completeness=0.0,
        system="none",
        n_measurements=0,
    )
    for grp, rkey in _TS_SYSTEM_PRIORITY:
        if grp not in store:
            continue
        g = store[grp]
        if "pe" not in g or "time" not in g or rkey not in g:
            continue
        try:
            pe = np.asarray(g["pe"], dtype=np.float64)  # (Tm, R)
            t_meas = np.asarray(g["time"], dtype=np.float64)  # (Tm,)
            rad = np.asarray(g[rkey], dtype=np.float64)  # (R,) or (Tm, R)
        except Exception:  # noqa: BLE001
            continue
        if pe.ndim != 2 or t_meas.ndim != 1 or pe.shape[0] != t_meas.shape[0]:
            continue

        # per-measurement features at valid times
        meas_t: list[float] = []
        meas_feat: list[np.ndarray] = []
        extents: list[float] = []
        for k in range(pe.shape[0]):
            r_row = rad[k] if rad.ndim == 2 else rad
            f = thomson_features_at_time(pe[k], r_row)
            if f[-1] > 0:  # fresh (had a usable profile)
                meas_t.append(float(t_meas[k]))
                meas_feat.append(f)
                extents.append(float(f[len(TS_R_GRID) + 4]))  # radial extent scalar
        if not meas_feat:
            continue

        meas_t_arr = np.asarray(meas_t)
        order = np.argsort(meas_t_arr)
        meas_t_arr = meas_t_arr[order]
        meas_feat_arr = np.asarray(meas_feat)[order]  # (M, 14)

        # forward-fill onto the engine grid: each grid time gets the most-recent
        # measurement AT OR BEFORE it (held at native cadence — NOT densified).
        out = np.zeros((T, N_THOMSON_FEATURES), dtype=np.float64)
        idx = np.searchsorted(meas_t_arr, grid_times, side="right") - 1
        valid = idx >= 0
        out[valid] = meas_feat_arr[idx[valid]]
        # freshness decays from the last measurement time (so held≠fresh)
        dt_since = np.full(T, np.inf)
        dt_since[valid] = grid_times[valid] - meas_t_arr[idx[valid]]
        out[:, -1] = np.where(valid, np.exp(-dt_since / _TS_FRESH_TAU_S), 0.0)

        completeness = _profile_completeness(float(np.median(extents)))
        logger.debug(
            "[ts] system=%s M=%d completeness=%.2f", grp, len(meas_feat), completeness
        )
        return ThomsonStream(
            features=out,
            completeness=completeness,
            system=grp,
            n_measurements=len(meas_feat),
        )
    return empty


# ---------------------------------------------------------------------------
# Integrated feature schema + per-shot loader (mag+ane + Thomson features)
# ---------------------------------------------------------------------------


def integrated_feature_schema() -> dict[str, list[str]]:
    """The widened input schema: mag+ane (v2) + the 14 Thomson pressure features.

    The Thomson block is appended LAST so the mag+ane column layout — and hence
    the GS grounding head's amb/amc slice map (``gs.grounding._feature_offsets``)
    — is UNCHANGED.  The grounding operator reconstructs raw magnetics from the
    same mag columns; the Thomson features only enrich what the ENCODER sees.
    """
    from imas_ambix.statespace.baseline import _FEATURE_SCHEMA_MAG_ANE  # noqa: PLC0415

    schema = dict(_FEATURE_SCHEMA_MAG_ANE)  # ama, amb, amc, ane (ordered)
    schema["thomson_pe"] = list(THOMSON_FEATURE_CHANNELS)
    return schema


def load_shot_integrated(
    shot_id: int,
    level1_dir: Path,
    target_channels: list[str],
    model_hz: float = 1000.0,
):
    """Load one shot's (X, y, times, plasma_on) with the integrated input set.

    X = [mag+ane columns (122) | Thomson pressure features (14)].  The mag+ane
    block is produced by the SAME ``baseline.load_shot_slices`` (so its columns,
    plasma-on masking, NaN imputation and target are bit-identical to v2); the
    Thomson block is forward-filled onto the same grid and concatenated.

    Returns ``(X, y, times, plasma_on, completeness, system)`` or None.  The
    per-shot ``completeness`` weights the (future) pressure likelihood and is
    persisted in the run cache so the grounding can use it; ``system`` records
    which TS system supplied the features.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _FEATURE_SCHEMA_MAG_ANE,
        load_shot_slices,
    )

    base = load_shot_slices(
        shot_id,
        _FEATURE_SCHEMA_MAG_ANE,
        target_channels,
        level1_dir=level1_dir,
        model_hz=model_hz,
        max_slices=None,
    )
    if base is None:
        return None
    X_mag, y, times, plasma_on = base

    shot_path = Path(level1_dir) / f"{shot_id}.zarr"
    try:
        store = zarr.open_group(str(shot_path), mode="r")
    except Exception:  # noqa: BLE001
        return None
    ts = load_thomson_stream(store, np.asarray(times, dtype=np.float64))

    X = np.concatenate([X_mag, ts.features.astype(X_mag.dtype)], axis=1)
    return X, y, times, plasma_on, ts.completeness, ts.system


# ---------------------------------------------------------------------------
# Split builder for the locked integrated input set (persisted artifact)
# ---------------------------------------------------------------------------


@dataclass
class IntegratedSplitResult:
    n_train: int
    n_cal: int
    n_ood: int
    n_total: int
    combo_key: str
    path: str
    ts_system_counts: dict = field(default_factory=dict)


def build_integrated_split(
    combo_key: str = "core_internal",
    *,
    restrict_gs_envelope: bool = True,
    cal_fraction: float = 0.12,
    seed: int = 42,
    output: Path | None = None,
) -> IntegratedSplitResult:
    """Build + persist the locked integrated-input split (the v0 OOD box).

    Reuses ``gs.integrated_inventory`` co-availability + the persisted regime
    scalars + the LOCKED v0 OOD box (joint_p84, by-current-density) so the OOD
    hole + cal set match T9 and stay comparable with v0.  Persisted via
    ``splits.ShotSplits.save``.
    """
    from imas_ambix.gs.integrated_inventory import (  # noqa: PLC0415
        _combo_shots,
        _load_inventory,
        _load_locked_ood_box,
        _load_regime_scalars,
        default_combos,
    )
    from imas_ambix.statespace.splits import build_splits  # noqa: PLC0415

    inv = _load_inventory()
    rs = _load_regime_scalars()
    box = _load_locked_ood_box()
    combos = {c.key: c for c in default_combos()}
    if combo_key not in combos:
        raise ValueError(f"unknown combo {combo_key!r}; have {sorted(combos)}")
    combo = combos[combo_key]
    shots = _combo_shots(inv, combo, restrict_gs_envelope=restrict_gs_envelope)
    sp = build_splits(
        shots,
        rs,
        held_out_family="dalpha",
        input_groups=[*combo.require_groups, "|".join(combo.any_of)],
        ood_box=box,
        cal_fraction=cal_fraction,
        seed=seed,
    )
    sp.notes.append(
        f"integrated-input-set={combo_key} (T10); restrict_gs_envelope="
        f"{restrict_gs_envelope}; LOCKED v0 OOD box (joint_p84, by-current-density)."
    )
    sp.notes.append(
        "T10 CORPUS FINDING: ams (MSE) in level-1 is the STATIC a-coefficient "
        "geometry table (acoeff shape [1,6,N], size-1 time dim) — NOT a "
        "time-resolved internal-current signal. The MSE→current/q half of "
        "internal-core is UNREALIZABLE from this corpus; only the Thomson→p′ "
        "half is wired (pressure-only internal grounding). See "
        "statespace/integrated_inputs.py docstring + the T10 verdict."
    )
    if output is None:
        output = Path("/work/projects/imas_gpu/mast/manifests") / (
            f"statespace_splits_integrated_{combo_key}_v0.json"
        )
    sp.save(output)
    return IntegratedSplitResult(
        n_train=sp.n_train,
        n_cal=sp.n_cal,
        n_ood=sp.n_ood,
        n_total=sp.n_total,
        combo_key=combo_key,
        path=str(output),
    )


def main(argv: list[str] | None = None) -> None:
    import argparse  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    p = argparse.ArgumentParser(
        description="Build the locked integrated-input split (T10)"
    )
    p.add_argument("--combo", default="core_internal")
    p.add_argument("--cal-fraction", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gs-envelope", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args(argv)
    res = build_integrated_split(
        a.combo,
        restrict_gs_envelope=not a.no_gs_envelope,
        cal_fraction=a.cal_fraction,
        seed=a.seed,
        output=a.output,
    )
    print(
        f"integrated split [{res.combo_key}]: train={res.n_train} cal={res.n_cal} "
        f"ood={res.n_ood} total={res.n_total} → {res.path}"
    )


if __name__ == "__main__":
    main()
