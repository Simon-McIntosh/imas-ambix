#!/usr/bin/env python
"""Feasibility oracle: can MEASURED DIAGNOSTICS predict plasma geometry?

The decisive follow-up to the camera-only oracle.  The camera oracle FAILED on
the interior geometry (magnetic-axis / X-point skill ~0 — the interior current
distribution is not a camera observable), but the MEASURED magnetics (flux loops
+ B-field probes) are precisely the inputs an EFIT-class reconstruction uses to
*determine* the boundary / X-point.  So a diagnostics→equilibrium referee should
be feasible where the camera one was not — and because the joint world model
already DREAMS these diagnostics, a feasible referee here lets the
controllability gate score the DREAMED diagnostics through this same map.

This oracle mirrors :mod:`scripts.feasibility_equilibrium_oracle` but swaps the
INPUT: instead of decoding camera tokens to pixels, it reads the per-window
MEASURED-SIGNAL tokens (the same magnetics / interferometer / soft-x-ray /
Dα-boundary / xsx / xim / ait / summary / pf_active / gas_injection streams the
WM conditions on) and trains a small temporal probe to predict the 12-D
equilibrium geometry.

EVALUATOR-ONLY (binding firewall)
---------------------------------
A third-party EVALUATOR.  The probe input is measured diagnostics; the LABEL is
the L2 equilibrium.  Nothing here is, or is importable by, the world-model
training path — the probe + labels only consume data and produce evaluator
metrics.  No WM checkpoint is loaded.  Equilibrium is an evaluator label only.

What it does
------------
1.  Build a shot-disjoint split (:mod:`imas_ambix.camdyn.splits`) over rbb-token
    shots, FORCING the controllability gate cohort + the standing held-out shots
    into the oracle TEST set (so the oracle is read on the SAME held-out plasma
    the gate is scored on, and never trains on it).
2.  Sample ~150-300 TRAIN shots; assemble one camera window (~0.25 s horizon) per
    shot for TRAIN and TEST to define the window's time span.
3.  Read the per-window MEASURED-SIGNAL tokens at ``n_signal_steps`` temporal
    positions across that span (NO camera decode — much cheaper).
4.  Build the 12-D equilibrium labels at the signal-grid times
    (:mod:`imas_ambix.worldmodel.equilibrium_labels`); the probe predicts the
    window-CENTRE geometry.
5.  Train the diagnostics probe
    (:mod:`imas_ambix.worldmodel.diagnostics_equilibrium_probe`) a few epochs on
    TRAIN; evaluate on TEST.
6.  Report PER-COMPONENT RMSE in METRES + the predict-the-TRAIN-mean baseline
    (the shot-to-shot spread) + skill = 1 - rmse/baseline, especially for
    axis_R, axis_Z, xpt_R, xpt_Z.  ABLATION: magnetics-only vs all-diagnostics.

Outputs (JSON + a pred-vs-true axis/X-point scatter) under
``/work/projects/imas_gpu/worldmodel/diagnostics_equilibrium_oracle/`` and
``docs/figures/joint-multimodal-plasma-wm/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("diag_eq_oracle")

# --- Output locations -------------------------------------------------------

DEFAULT_OUT_ROOT = Path(
    "/work/projects/imas_gpu/worldmodel/diagnostics_equilibrium_oracle"
)
DEFAULT_FIG_DIR = Path("docs/figures/joint-multimodal-plasma-wm")

# --- Cohorts that MUST be in the oracle TEST set (never trained on) ----------
#: The controllability gate cohort + the standing held-out shots.  Forced into
#: the oracle TEST partition so the feasibility verdict is read on the SAME
#: plasma the downstream gate scores.
GATE_COHORT = (15089, 15223, 15517, 15963, 15972, 16024, 16223)
STANDING_HELD_OUT = (18502, 18503, 18504, 18505)
FORCED_TEST_SHOTS = tuple(sorted(set(GATE_COHORT) | set(STANDING_HELD_OUT)))

#: The synthetic stream name for the L2 ``pf_active`` coil currents the oracle
#: reads directly (no staged store exists).  Coil currents are an EFIT input —
#: the boundary / X-point is placed from {magnetics + COIL CURRENTS + Ip +
#: machine geometry} — so this stream joins the headline EFIT-input arm.
PF_ACTIVE_STREAM = "pf_active_coils"

#: The synthetic stream name for the L2 toroidal saddle-loop voltage array (12
#: channels at distinct toroidal angles φ) the oracle reads directly.  The ONE
#: ingestible toroidal field series in this dataset; with periodic-φ PE it lets
#: the model resolve toroidal structure the single-φ poloidal arrays cannot.
SADDLE_STREAM = "toroidal_saddle"

#: Streams in the headline EFIT-input arm: ``magnetics`` (calibrated L2 flux
#: loops + B-field probes + the device-global plasma current Ip), ``xma`` (the
#: HF magnetics codebook), the PF-active COIL CURRENTS and the toroidal SADDLE
#: array.  Together these are the EFIT-class {position/shape sensor + Ip +
#: current actuators + toroidal array} set.
MAGNETICS_STREAMS = ("xma", "magnetics", PF_ACTIVE_STREAM, SADDLE_STREAM)

#: The staged group whose column names resolve to L2-IDS apparatus geometry
#: directly (the EFIT-class position/shape sensor).  Every other stream's
#: per-channel geometry comes from the campaign table via
#: :func:`imas_ambix.tokenizer.geometry_reader.geometry_for_channels`.
_MAGNETICS_GROUP = "magnetics"

#: L2 ``pf_active`` raw current arrays (signed, physical units) the coil stream
#: reads — coil currents + the solenoid; named 1:1 by ``current_channel``.
_PF_ACTIVE_CURRENT_KEYS = ("coil_current", "solenoid_current")


def read_pf_active_coils(shot_id, grid, *, calibration=None):
    """Read L2 ``pf_active`` coil currents -> (ids, values, names, geometry, kinds).

    The EFIT current actuators.  Reads the L2 ``pf_active`` group's signed
    ``coil_current`` (n_coil, n_time) arrays (named 1:1 by ``current_channel``),
    resamples to the window ``grid``, corpus-standardises each channel (SIGNED;
    no abs) and ALSO quantises to the L2 256-bin local ids for the token-lane
    comparison.  Geometry is each coil's filament-centroid ``(R, Z)`` via
    :func:`imas_ambix.tokenizer.geometry_reader.pf_active_geometry_for_channels`.

    Returns ``(ids (S, C) int64, values (S, C) float32, names, geom (C, 10)
    float32, kinds tuple)`` or ``None`` when the L2 group is unreadable.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415
    from imas_ambix.statespace.align import align_chord2d_to_grid  # noqa: PLC0415
    from imas_ambix.tokenizer.geometry_reader import (  # noqa: PLC0415
        pf_active_geometry_for_channels,
    )
    from imas_ambix.worldmodel.spacetime_dataset_v2 import _quantise_l2  # noqa: PLC0415

    path = LEVEL2_DIR / f"{int(shot_id)}.zarr"
    if not path.exists():
        return None
    try:
        root = zarr.open_group(str(path), mode="r")
        if "pf_active" not in set(root.group_keys()):
            return None
        grp = root["pf_active"]
        keys = set(grp.array_keys())
        if "time" not in keys:
            # pf_active time axis is named in current_channel's coordinate; the
            # coil arrays are (n_coil, n_time) — find the time length from them.
            pass
        # locate a usable time axis (1-D, matches the current arrays' 2nd dim).
        tkey = next((k for k in ("time", "coil_current_time") if k in keys), None)
        cols, names = [], []
        for ckey in _PF_ACTIVE_CURRENT_KEYS:
            if ckey not in keys:
                continue
            arr = np.asarray(grp[ckey], dtype=np.float64)
            if arr.ndim != 2:
                continue
            # orient to (n_time, n_coil).
            chname_key = "current_channel" if ckey == "coil_current" else None
            chnames = (
                [str(x) for x in np.asarray(grp[chname_key]).reshape(-1)]
                if (chname_key and chname_key in keys)
                else [f"{ckey}[{i}]" for i in range(min(arr.shape))]
            )
            n_coil = len(chnames)
            arr2 = arr if arr.shape[1] == n_coil else arr.T
            cols.append(arr2)
            names.extend(chnames)
        if not cols:
            return None
        raw = np.concatenate(cols, axis=1)  # (n_time, C)
        if tkey is not None:
            vtime = np.asarray(grp[tkey], dtype=np.float64).reshape(-1)
        else:
            # fall back to a uniform time over the array length (rare).
            vtime = np.linspace(grid[0], grid[-1], raw.shape[0], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None
    on_grid = align_chord2d_to_grid(raw, vtime, grid).astype(np.float64)  # (S, C)
    values = _standardise_continuous(on_grid, names, calibration).astype(np.float32)
    ids = _quantise_l2(on_grid, channel_names=names, calibration=calibration).astype(
        np.int64
    )
    ag = pf_active_geometry_for_channels(names, int(shot_id))
    return (
        ids,
        values,
        names,
        np.asarray(ag.features, dtype=np.float32),
        tuple(ag.sensor_kinds),
    )


def read_saddle_toroidal(shot_id, grid, *, calibration=None):
    """Read L2 toroidal saddle voltages -> (ids, values, names, geometry, kinds).

    The ONE ingestible toroidal field series: ``b_field_tor_probe_saddle_voltage``
    (12 channels at 12 distinct toroidal angles φ on the faster ``time_saddle``
    base).  Reads the signed voltages, resamples to the window ``grid`` (the array
    is multi-rate — handled by the grid resampler), corpus-standardises (SIGNED)
    and quantises to L2 ids; geometry is each loop's ``(R, Z, φ)`` via
    :func:`imas_ambix.tokenizer.geometry_reader.saddle_toroidal_geometry`.  φ is
    encoded periodically at the model input so the toroidal seam is continuous.

    Returns ``(ids, values, names, geom (C,10), kinds)`` or ``None`` when absent.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415
    from imas_ambix.statespace.align import align_chord2d_to_grid  # noqa: PLC0415
    from imas_ambix.tokenizer.geometry_reader import (  # noqa: PLC0415
        saddle_toroidal_geometry,
    )
    from imas_ambix.worldmodel.spacetime_dataset_v2 import _quantise_l2  # noqa: PLC0415

    path = LEVEL2_DIR / f"{int(shot_id)}.zarr"
    if not path.exists():
        return None
    try:
        grp = zarr.open_group(str(path), mode="r")["magnetics"]
        keys = set(grp.array_keys())
        vkey = "b_field_tor_probe_saddle_voltage"
        if vkey not in keys or "time_saddle" not in keys:
            return None
        volt = np.asarray(grp[vkey], dtype=np.float64)  # (n_coil, n_time)
        names = (
            [str(x) for x in np.asarray(grp[f"{vkey}_channel"]).reshape(-1)]
            if f"{vkey}_channel" in keys
            else [f"{vkey}[{i}]" for i in range(volt.shape[0])]
        )
        vtime = np.asarray(grp["time_saddle"], dtype=np.float64).reshape(-1)
        raw = volt.T if volt.shape[1] == vtime.shape[0] else volt  # (n_time, C)
    except Exception:  # noqa: BLE001
        return None
    on_grid = align_chord2d_to_grid(raw, vtime, grid).astype(np.float64)  # (S, C)
    values = _standardise_continuous(on_grid, names, calibration).astype(np.float32)
    ids = _quantise_l2(on_grid, channel_names=names, calibration=calibration).astype(
        np.int64
    )
    src = saddle_toroidal_geometry(int(shot_id))
    if src is None:
        return None
    _src_names, feats, kinds = src
    return ids, values, names, np.asarray(feats, dtype=np.float32), tuple(kinds)


def stream_channel_names(shot_id, modalities, *, token_root=None):
    """Return ``{stream_name: (channel_name, ...)}`` for a shot's present streams.

    Reads the DETERMINISTIC per-channel names every stream is keyed by — the
    staged groups via :func:`_read_staged_raw` (column names
    ``b_field_pol_probe_{kind}_field[i]`` / ``flux_loop_flux[i]`` / ``ip`` …),
    the pre-tokenised ``signal_hf`` groups via the store's ``channel_names``
    attribute.  A stream whose store is unreadable is omitted.  These names feed
    the per-sensor geometry lookup so each token carries its sensor's geometry.
    """
    from imas_ambix.worldmodel.dataset import _read_signal_hf
    from imas_ambix.worldmodel.spacetime_dataset_v2 import _read_staged_raw

    out = {}
    for m in modalities:
        try:
            if m.kind in ("ait", "staged"):
                _raw, names, _t = _read_staged_raw(
                    m.group,
                    int(shot_id),
                    token_root=token_root,
                    profile_r_stride=m.profile_r_stride,
                )
            else:
                _tok, _t, _v, names, _b = _read_signal_hf(
                    int(shot_id), m.group, token_root=token_root
                )
        except FileNotFoundError, KeyError, OSError:
            continue
        out[m.name] = tuple(str(c) for c in names)
    return out


def stream_geometry(shot_id, modalities, names_by_stream):
    """Resolve per-sensor geometry for every present stream of a shot.

    Returns ``{stream_name: (features (C, 10) float32, kinds (C,) str)}`` aligned
    1:1 with the stream's channel names.  The ``magnetics`` group resolves via
    the L2-IDS apparatus geometry (``b_field_pol_probe_*`` / ``flux_loop_*``
    column index -> sensor R/Z + orientation); every other stream resolves via
    the campaign geometry table (``ip``, coils, chords, scalars present + explicit
    with NaN coordinates) — never a silent geometry drop.
    """
    from imas_ambix.tokenizer.geometry_reader import (
        geometry_for_channels,
        l2_signal_hf_geometry_for_channels,
        magnetics_geometry_for_channels,
    )

    fields = _campaign_geometry_fields(int(shot_id), names_by_stream)
    out = {}
    group_by_stream = {m.name: m.group for m in modalities}
    for name, channel_names in names_by_stream.items():
        group = group_by_stream.get(name)
        if group == _MAGNETICS_GROUP:
            ag = magnetics_geometry_for_channels(channel_names, int(shot_id))
        elif channel_names and "." in str(channel_names[0]):
            # L2 light-path signal_hf streams name channels "{group}.{var}[i]"
            # (soft_x_rays cameras -> chord geometry, pf_active -> coil, etc.) —
            # route them through the all-signals consolidation resolver so every
            # stream carries its apparatus geometry, not an all-NaN scalar block.
            ag = l2_signal_hf_geometry_for_channels(channel_names, int(shot_id))
        else:
            ag = geometry_for_channels(channel_names, fields=fields)
        out[name] = (
            np.asarray(ag.features, dtype=np.float32),
            tuple(ag.sensor_kinds),
        )
    return out


def _campaign_geometry_fields(shot_id, names_by_stream):
    """Build the campaign geometry table for a shot (None if unavailable).

    Seeds the table with EVERY non-magnetics channel name so coils / chords /
    scalars are present + explicitly kinded (NaN coordinates), never dropped.
    The magnetics group does NOT route through this table — it reads the L2 IDS
    geometry directly.
    """
    from imas_ambix.gs.geometry_export import build_geometry_table

    extra = []
    for name, channel_names in names_by_stream.items():
        if name == _MAGNETICS_GROUP:
            continue
        extra.extend(channel_names)
    try:
        return build_geometry_table(int(shot_id), extra_channel_names=extra)
    except FileNotFoundError, KeyError, OSError, ValueError:
        return None


def continuous_stream_values(
    shot_id,
    modalities,
    names_by_stream,
    grid,
    *,
    token_root=None,
    calibration_by_group=None,
):
    """Continuous, corpus-standardised per-sensor values for the staged groups.

    Reads the RAW staged floats (``b_field_pol_probe_*`` / ``flux_loop_*`` / …),
    resamples to the window ``grid`` and standardises each channel by its CORPUS
    mean/std (the same absolute scale the magnetics ceiling uses).  Returns
    ``{stream_name: (n_steps, n_channels) float32}`` aligned 1:1 with the
    stream's channel names; only the staged groups (which carry raw floats here)
    are populated, so the quantisation ablation is exercised exactly where the
    magnetics ceiling is — a stream with no raw-float source is simply absent
    from the continuous dict (the probe then needs the id path for it).
    """
    from imas_ambix.statespace.align import align_chord2d_to_grid
    from imas_ambix.worldmodel.spacetime_dataset_v2 import _read_staged_raw

    out = {}
    group_by_stream = {m.name: m for m in modalities}
    for name in names_by_stream:
        m = group_by_stream.get(name)
        if m is None or m.kind not in ("staged", "ait"):
            continue
        try:
            raw, names, vtime = _read_staged_raw(
                m.group,
                int(shot_id),
                token_root=token_root,
                profile_r_stride=m.profile_r_stride,
            )
        except FileNotFoundError, KeyError, OSError:
            continue
        cap = int(m.max_channels)
        raw = raw[:, :cap]
        names = list(names[:cap])
        on_grid = align_chord2d_to_grid(raw, vtime, grid).astype(np.float64)  # (S, C)
        cal = (
            calibration_by_group.get(m.group)
            if calibration_by_group is not None
            else None
        )
        z = _standardise_continuous(on_grid, names, cal)
        out[name] = z.astype(np.float32)
    return out


def _standardise_continuous(on_grid, names, cal):
    """Corpus-standardise a ``(S, C)`` raw block per channel (NaN -> 0).

    Uses the corpus calibration mean/std when present (absolute scale preserved);
    falls back to a per-window z-score per channel when no calibration is on
    disk, so the continuous arm still has a well-conditioned input.
    """
    s, c = on_grid.shape
    z = np.zeros((s, c), dtype=np.float64)
    for j in range(c):
        col = on_grid[:, j]
        fin = np.isfinite(col)
        if cal is not None and j < len(names) and names[j] in cal:
            cc = cal[names[j]]
            mu = float(cc.mean)
            sd = float(cc.std) or 1.0
        elif fin.sum() > 1:
            mu = float(np.mean(col[fin]))
            sd = float(np.std(col[fin])) or 1.0
        else:
            mu, sd = 0.0, 1.0
        zc = np.where(fin, (col - mu) / sd, 0.0)
        z[:, j] = zc
    return z


def machine_geometry_points(shot_id, *, max_points=8):
    """A few ``(R, Z, is_coil)`` machine-geometry context points for a shot.

    The vessel-contour extent corners + a sample of PF-coil centroids, from the
    campaign :class:`MachineGeometry` block — the fixed machine frame every
    channel shares.  Returns ``(M, 3) float32`` (empty when unavailable).
    """
    from imas_ambix.gs.geometry_export import build_geometry_table

    try:
        fields = build_geometry_table(int(shot_id))
    except FileNotFoundError, KeyError, OSError, ValueError:
        return np.zeros((0, 3), dtype=np.float32)
    mach = fields.machine
    pts = []
    lr = np.asarray(mach.limiter_r, dtype=np.float64)
    lz = np.asarray(mach.limiter_z, dtype=np.float64)
    if lr.size and lz.size:
        # vessel-extent corners (R/Z min/max) — the machine bounding frame.
        pts.append((float(lr.min()), float(lz.min()), 0.0))
        pts.append((float(lr.max()), float(lz.max()), 0.0))
        pts.append((float(lr.min()), float(lz.max()), 0.0))
        pts.append((float(lr.max()), float(lz.min()), 0.0))
    pr = np.asarray(mach.pf_coil_r, dtype=np.float64)
    pz = np.asarray(mach.pf_coil_z, dtype=np.float64)
    budget = max_points - len(pts)
    if pr.size and pz.size and budget > 0:
        stride = max(1, pr.size // budget)
        for j in range(0, pr.size, stride):
            if len(pts) >= max_points:
                break
            pts.append((float(pr[j]), float(pz[j]), 1.0))
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)


# ---------------------------------------------------------------------------
# Window + signal + label assembly (one labelled example per shot window)
# ---------------------------------------------------------------------------


def _select_brightest_start(token_path, level1_path, camera, config):
    """Pick the brightest valid window start for a shot (most plasma-active).

    Returns an int start frame, or None to fall back to the centred window.
    Brightness ranks candidate starts by mean raw-frame intensity (the honest
    activity proxy) so the probe sees an established plasma, not a dark ramp.
    """
    try:
        from imas_ambix.camdyn.reconstruction_demo import _window_brightness
        from imas_ambix.worldmodel.spacetime_dataset import (
            _fps_from_times,
            _frame_times,
            camera_frame_count,
            effective_frame_stride,
        )
    except Exception:  # noqa: BLE001
        return None
    try:
        shot_id = int(Path(token_path).parent.name)
        times = _frame_times(shot_id, camera, token_root=None)
        n_total = camera_frame_count(shot_id, camera, token_root=None)
    except Exception:  # noqa: BLE001
        return None
    fps = _fps_from_times(times)
    stride = effective_frame_stride(config, fps)
    span = (config.n_frames - 1) * stride + 1
    if n_total < span:
        return None
    # candidate starts every span//2 frames
    step = max(1, span // 2)
    starts = list(range(0, n_total - span + 1, step))
    if not starts:
        return None
    bright = _window_brightness(shot_id, starts, span)
    if bright is None:
        return None
    return int(starts[int(np.argmax(bright))])


def assemble_examples(
    shot_ids,
    *,
    camera,
    modalities,
    n_signal_steps,
    config,
    level2_root,
    token_root,
    calibration_by_group=None,
):
    """Build one labelled diagnostics example per shot.

    For each shot: assemble a camera window (brightest start, fall back centred)
    to fix the ~0.25 s time span; read the measured-signal tokens at
    ``n_signal_steps`` positions across that span; build the 12-D equilibrium
    labels at those signal-grid times; keep the WINDOW-CENTRE label as the target.

    Returns a list of dicts: ``{shot_id, signals {name:(S,C)}, target (12,),
    mask (12,)}`` for every shot that yields a window with >=1 present stream and
    >=1 finite label component at the centre.
    """
    from imas_ambix.camdyn.dataset import discover_token_shots
    from imas_ambix.worldmodel.equilibrium_labels import load_equilibrium_geometry
    from imas_ambix.worldmodel.spacetime_dataset import assemble_window
    from imas_ambix.worldmodel.spacetime_dataset_v2 import read_window_signals

    specs = discover_token_shots(
        camera=camera,
        token_root=token_root,
        shot_ids=list(shot_ids),
        read_n_frames=False,
    )
    spec_by_shot = {s.shot_id: s for s in specs}

    out = []
    for sid in shot_ids:
        spec = spec_by_shot.get(int(sid))
        if spec is None:
            logger.info("shot %d: no rbb tokens — skip", sid)
            continue
        start = _select_brightest_start(
            spec.token_path, spec.level1_path, camera, config
        )
        try:
            sample = assemble_window(
                int(sid),
                config,
                camera=camera,
                token_root=token_root,
                start_frame=start,
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.info("shot %d: no window (%s) — skip", sid, exc)
            continue
        signals = read_window_signals(
            int(sid),
            sample,
            modalities,
            n_signal_steps,
            token_root=token_root,
            calibration_by_group=calibration_by_group,
        )
        if not signals:
            logger.info("shot %d: no readable measured streams — skip", sid)
            continue
        # per-sensor geometry for every present stream (the positional code) +
        # the shared machine-frame context points.
        names_by_stream = stream_channel_names(
            int(sid), modalities, token_root=token_root
        )
        names_by_stream = {k: v for k, v in names_by_stream.items() if k in signals}
        geom_by_stream = stream_geometry(int(sid), modalities, names_by_stream)
        machine_pts = machine_geometry_points(int(sid))
        # CONTINUOUS standardised per-sensor values (the quantisation ablation):
        # raw staged floats resampled to the SAME grid + corpus-standardised, so
        # a continuous-value arm can replace the 256-bin token id with the
        # unquantised reading.  Only the staged groups carry raw floats here.
        grid_for_values = np.linspace(
            float(np.asarray(sample.frame_time).min()),
            float(np.asarray(sample.frame_time).max()),
            int(n_signal_steps),
            dtype=np.float64,
        )
        values_by_stream = continuous_stream_values(
            int(sid),
            modalities,
            names_by_stream,
            grid_for_values,
            token_root=token_root,
            calibration_by_group=calibration_by_group,
        )
        # PF-active COIL CURRENTS read directly from L2 (no staged store): an
        # EFIT current actuator the boundary / X-point is placed from.  Signed,
        # corpus-standardised continuous values + the token-id lane + per-coil
        # geometry.  Joins the headline EFIT-input stream set.
        pf_cal = (
            calibration_by_group.get("pf_active")
            if calibration_by_group is not None
            else None
        )
        pf = read_pf_active_coils(int(sid), grid_for_values, calibration=pf_cal)
        if pf is not None:
            pf_ids, pf_vals, _pf_names, pf_geom, pf_kinds = pf
            signals[PF_ACTIVE_STREAM] = np.asarray(pf_ids, np.int64)
            values_by_stream[PF_ACTIVE_STREAM] = np.asarray(pf_vals, np.float32)
            geom_by_stream[PF_ACTIVE_STREAM] = (pf_geom, pf_kinds)
        # Toroidal SADDLE array read directly from L2 (no staged store): 12 loops
        # at distinct toroidal angles φ — the one ingestible toroidal field
        # series, with periodic-φ geometry so the model resolves toroidal
        # structure the single-φ poloidal arrays cannot.  Signed continuous
        # values + the token-id lane + per-loop (R, Z, φ) geometry.
        sad_cal = (
            calibration_by_group.get("magnetics")
            if calibration_by_group is not None
            else None
        )
        sad = read_saddle_toroidal(int(sid), grid_for_values, calibration=sad_cal)
        if sad is not None:
            sad_ids, sad_vals, _sad_names, sad_geom, sad_kinds = sad
            signals[SADDLE_STREAM] = np.asarray(sad_ids, np.int64)
            values_by_stream[SADDLE_STREAM] = np.asarray(sad_vals, np.float32)
            geom_by_stream[SADDLE_STREAM] = (sad_geom, sad_kinds)
        # equilibrium labels on the SIGNAL grid (same span as the window).
        ftime = np.asarray(sample.frame_time, dtype=np.float64)
        t0, t1 = float(ftime.min()), float(ftime.max())
        grid = np.linspace(t0, t1, int(n_signal_steps), dtype=np.float64)
        try:
            geo = load_equilibrium_geometry(int(sid), grid, level2_root=level2_root)
        except (KeyError, FileNotFoundError) as exc:
            logger.info("shot %d: no equilibrium (%s) — skip", sid, exc)
            continue
        # window-CENTRE label (the probe predicts the geometry at mid-window).
        cidx = int(n_signal_steps // 2)
        tgt = geo.target[cidx]  # (12,)
        msk = geo.finite_mask[cidx]  # (12,)
        if not msk.any():
            # centre is masked (off-plasma) — try the nearest finite step.
            any_finite = geo.finite_mask.any(axis=1)
            if not any_finite.any():
                logger.info("shot %d: all-masked equilibrium window — skip", sid)
                continue
            order = np.argsort(np.abs(np.arange(n_signal_steps) - cidx))
            for j in order:
                if any_finite[j]:
                    tgt = geo.target[j]
                    msk = geo.finite_mask[j]
                    break
        out.append(
            {
                "shot_id": int(sid),
                "signals": {k: np.asarray(v, np.int64) for k, v in signals.items()},
                "geometry": {k: g[0] for k, g in geom_by_stream.items()},
                "sensor_kinds": {k: g[1] for k, g in geom_by_stream.items()},
                "values": values_by_stream,
                "machine": np.asarray(machine_pts, np.float32),
                "target": np.asarray(tgt, np.float32),
                "mask": np.asarray(msk, bool),
            }
        )
        logger.info(
            "shot %d: streams=%s  finite-comp=%d/12",
            sid,
            ",".join(sorted(signals)),
            int(msk.sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Stream sizing + tensor batching
# ---------------------------------------------------------------------------


#: Vocab for the synthetic PF-active coil-current stream (L2 256-bin local ids).
def _pf_active_vocab():
    from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB

    return int(L2_BLOCK_VOCAB) + 1


#: Synthetic streams read directly from L2 (not modalities): name -> channel cap.
#: PF-active is 10 coils + solenoid; the toroidal saddle array is 12 loops.
_SYNTHETIC_STREAMS = {PF_ACTIVE_STREAM: 16, SADDLE_STREAM: 16}


def probe_channels(examples, modalities):
    """Max channel count seen per stream across the assembled examples.

    Caps at each modality's ``max_channels`` (and each synthetic stream's own
    cap).  A stream never present keeps 0 and is dropped from the model's stream
    list.
    """
    cap = {m.name: int(m.max_channels) for m in modalities}
    cap.update(_SYNTHETIC_STREAMS)
    seen = {m.name: 0 for m in modalities}
    for name in _SYNTHETIC_STREAMS:
        seen[name] = 0
    for ex in examples:
        for name, arr in ex["signals"].items():
            if name in seen:
                seen[name] = max(seen[name], int(arr.shape[1]))
    return {k: min(v, cap.get(k, v)) for k, v in seen.items()}


def build_stream_specs(channels, modalities, *, restrict=None):
    """Build probe :class:`StreamSpec` list from probed channels.

    ``restrict`` (optional set of stream names) keeps only those streams — the
    ablation lever.  A stream with 0 probed channels is dropped.  The synthetic
    streams (PF-active coil currents + the toroidal saddle array, read directly
    from L2, not modalities) are appended when present.
    """
    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import StreamSpec

    specs = []
    for m in modalities:
        if restrict is not None and m.name not in restrict:
            continue
        c = int(channels.get(m.name, 0))
        if c <= 0:
            continue
        specs.append(StreamSpec(name=m.name, vocab=int(m.vocab), channels=c))
    # synthetic streams (read directly from L2 — not modalities).
    for name in _SYNTHETIC_STREAMS:
        if restrict is not None and name not in restrict:
            continue
        c = int(channels.get(name, 0))
        if c > 0:
            specs.append(StreamSpec(name=name, vocab=_pf_active_vocab(), channels=c))
    return specs


def batch_signals(examples, specs, n_steps, *, device, continuous=False):
    """Stack examples into the probe's per-stream input tensors.

    Returns ``(signals, geometry, sensor_kinds, values, machine)``:

    * ``signals[stream]``   — ``(N, n_steps, channels) int64`` local ids
      (PAD-filled per stream so an example missing a stream is all-PAD and
      shapes are uniform);
    * ``geometry[stream]``  — ``(N, channels, 10) float32`` per-sensor geometry
      (NaN rows for channels with no known geometry / PAD lanes);
    * ``sensor_kinds[stream]`` — ``(N, channels) int64`` per-sensor kind indices;
    * ``values[stream]``    — ``(N, n_steps, channels) float32`` CONTINUOUS
      standardised values (only populated for streams that carry a raw-float
      source; used by the quantisation ablation) or ``None`` when ``continuous``
      is False;
    * ``machine``           — ``(N, M, 3) float32`` machine-frame context points
      (zero-padded to the corpus-max point count).

    Each stream is padded / truncated to its spec channel count, with geometry
    and kinds aligned to the SAME channel order so each token gets its sensor's
    geometry.
    """
    import torch

    from imas_ambix.worldmodel.dataset import PAD_LOCAL_ID
    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import (
        N_GEOM_FEATURES,
        sensor_kind_index,
    )

    n = len(examples)
    signals, geometry, kinds, values = {}, {}, {}, {}
    for sp in specs:
        ids = np.full((n, n_steps, sp.channels), PAD_LOCAL_ID, dtype=np.int64)
        geom = np.full((n, sp.channels, N_GEOM_FEATURES), np.nan, dtype=np.float32)
        kind = np.zeros((n, sp.channels), dtype=np.int64)  # 0 == "unknown"
        val = np.zeros((n, n_steps, sp.channels), dtype=np.float32)
        for i, ex in enumerate(examples):
            arr = ex["signals"].get(sp.name)
            if arr is None:
                continue
            s = min(n_steps, arr.shape[0])
            c = min(sp.channels, arr.shape[1])
            ids[i, :s, :c] = np.clip(arr[:s, :c], 0, sp.vocab - 1)
            g = ex.get("geometry", {}).get(sp.name)
            if g is not None:
                gc = min(sp.channels, g.shape[0])
                geom[i, :gc, :] = g[:gc, :]
            sk = ex.get("sensor_kinds", {}).get(sp.name)
            if sk is not None:
                for j in range(min(sp.channels, len(sk))):
                    kind[i, j] = sensor_kind_index(sk[j])
            if continuous:
                v = ex.get("values", {}).get(sp.name)
                if v is not None:
                    vs = min(n_steps, v.shape[0])
                    vc = min(sp.channels, v.shape[1])
                    val[i, :vs, :vc] = v[:vs, :vc]
        signals[sp.name] = torch.from_numpy(ids).to(device)
        geometry[sp.name] = torch.from_numpy(geom).to(device)
        kinds[sp.name] = torch.from_numpy(kind).to(device)
        values[sp.name] = torch.from_numpy(val).to(device)

    # machine-frame context points — zero-padded to the corpus-max count.
    m_max = max(
        (
            int(np.asarray(ex.get("machine", np.zeros((0, 3)))).shape[0])
            for ex in examples
        ),
        default=0,
    )
    mach = np.zeros((n, m_max, 3), dtype=np.float32)
    for i, ex in enumerate(examples):
        mp = np.asarray(ex.get("machine", np.zeros((0, 3))), dtype=np.float32)
        if mp.size:
            mc = min(m_max, mp.shape[0])
            mach[i, :mc, :] = mp[:mc, :]
    machine = torch.from_numpy(mach).to(device)

    return signals, geometry, kinds, (values if continuous else None), machine


# ---------------------------------------------------------------------------
# Standardisation + train + eval
# ---------------------------------------------------------------------------


def standardise_stats(y, mask):
    """Per-component mean / std over the finite TRAIN labels (NaN-safe)."""
    dim = y.shape[1]
    mean = np.zeros(dim)
    std = np.ones(dim)
    for d in range(dim):
        vals = y[mask[:, d], d]
        if vals.size > 1:
            mean[d] = float(np.mean(vals))
            std[d] = float(np.std(vals)) or 1.0
    return mean, std


def _slice_batch(batch, idx):
    """Index a ``(signals, geometry, kinds, values, machine)`` batch by ``idx``."""
    signals, geometry, kinds, values, machine = batch
    sg = {k: v[idx] for k, v in signals.items()}
    gm = {k: v[idx] for k, v in geometry.items()}
    kn = {k: v[idx] for k, v in kinds.items()}
    vl = {k: v[idx] for k, v in values.items()} if values is not None else None
    mc = machine[idx] if machine is not None else None
    return sg, gm, kn, vl, mc


def train_probe(
    tr_examples,
    specs,
    *,
    n_steps,
    target_dim,
    epochs,
    batch_size,
    lr,
    device,
    seed,
    continuous=False,
):
    """Train the space-time probe; returns (model, target_mean, target_std)."""
    import torch

    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import (
        DiagnosticsEquilibriumProbe,
        DiagnosticsProbeConfig,
        gaussian_nll,
        set_xpoint_loss,
    )
    from imas_ambix.worldmodel.equilibrium_labels import N_XPOINT_SLOTS

    # X-point null-set component span in the target vector (axis is 0,1; the set
    # is the next 2*N_XPOINT_SLOTS; LCFS follows).  Axis + LCFS use the masked
    # Gaussian NLL; the set uses the permutation-invariant matched loss.
    xpt_start = 2
    xpt_end = xpt_start + 2 * N_XPOINT_SLOTS

    torch.manual_seed(seed)
    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    mean, std = standardise_stats(ytr, mtr)
    ystd = (np.nan_to_num(ytr, nan=0.0) - mean) / std
    ystd = np.where(mtr, ystd, 0.0).astype(np.float32)

    dev = torch.device(device)
    batch = batch_signals(
        tr_examples, specs, n_steps, device=dev, continuous=continuous
    )
    y_t = torch.from_numpy(ystd).to(dev)
    m_t = torch.from_numpy(mtr.astype(np.float32)).to(dev)
    n = y_t.shape[0]

    # Overfit control: the small TRAIN set (~184 survivors) overfits a deep
    # probe, so keep the width modest + lean on weight decay.  A moderate dropout
    # (0.15) — heavier dropout (0.3) was measured to HURT axis_Z, so back it off.
    # stft_phase is ON (continuous lane) so cross-sensor phase reaches attention.
    cfg = DiagnosticsProbeConfig(
        streams=list(specs),
        n_steps=n_steps,
        target_dim=target_dim,
        continuous_value=continuous,
        d_model=160,
        n_heads=4,
        n_layers=4,
        dropout=0.15,
        stft_phase=True,
    )
    model = DiagnosticsEquilibriumProbe(cfg).to(dev)
    logger.info(
        "probe params: %.2fM  streams=%s  continuous=%s",
        model.n_parameters() / 1e6,
        [s.name for s in specs],
        continuous,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # A deep space-time transformer over hundreds of (sensor x time) tokens needs
    # a warmup + cosine schedule to converge (a flat LR underfits in a few-epoch
    # budget — the NLL was still falling at the cap).  Warmup over the first ~10%
    # of steps, cosine-decay to ~3% of the peak LR over the rest.
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.1 * total_steps))

    def _lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.03 + 0.97 * 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_scale)

    model.train()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size].to(dev)
            sb, gb, kb, vb, mcb = _slice_batch(batch, idx)
            yb = y_t[idx]
            mb = m_t[idx]
            opt.zero_grad()
            with torch.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")
            ):
                # one encoder pass -> the regression head + the presence head.
                pooled = model.pooled_embedding(sb, gb, kb, values=vb, machine=mcb)
                out = model.head(pooled)
                pmean, plog = out.chunk(2, dim=-1)
                pres = model.presence_head(pooled)
            pmean = pmean.float()
            plog = plog.float()
            # axis + LCFS: masked Gaussian NLL on the NON-set components only.
            non_set = mb.clone()
            non_set[:, xpt_start:xpt_end] = 0.0
            loss_axis_lcfs = gaussian_nll(pmean, plog, yb, non_set)
            # X-point null SET: permutation-invariant matched loss + presence BCE.
            loss_set = set_xpoint_loss(
                pmean,
                plog,
                pres.float(),
                yb,
                mb,
                xpoint_start=xpt_start,
                n_slots=N_XPOINT_SLOTS,
            )
            loss = loss_axis_lcfs + loss_set
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        logger.info(
            "epoch %d/%d  NLL=%.4f  lr=%.2e",
            ep + 1,
            epochs,
            tot / max(nb, 1),
            sched.get_last_lr()[0],
        )
    return model, mean, std


def evaluate(
    model,
    te_examples,
    specs,
    mean,
    std,
    *,
    n_steps,
    device,
    batch_size,
    continuous=False,
):
    """Predict on TEST -> (pred (n,D) metres, y, mask, presence_prob (n,slots))."""
    import torch

    dev = torch.device(device)
    batch = batch_signals(
        te_examples, specs, n_steps, device=dev, continuous=continuous
    )
    yte = np.stack([ex["target"] for ex in te_examples]).astype(np.float32)
    mte = np.stack([ex["mask"] for ex in te_examples]).astype(bool)
    n = yte.shape[0]
    model.eval()
    preds = []
    pres_probs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n), device=dev)
            sb, gb, kb, vb, mcb = _slice_batch(batch, idx)
            pooled = model.pooled_embedding(sb, gb, kb, values=vb, machine=mcb)
            out = model.head(pooled)
            pmean, _plog = out.chunk(2, dim=-1)
            mu = pmean.detach().cpu().float().numpy()
            preds.append(mu * np.asarray(std) + np.asarray(mean))
            pres = torch.sigmoid(model.presence_head(pooled)).detach().cpu().numpy()
            pres_probs.append(pres)
    pred = np.concatenate(preds, axis=0)
    presence_prob = np.concatenate(pres_probs, axis=0)
    return pred, yte, mte, presence_prob


def set_xpoint_metrics(pred, y, mask, presence_prob, *, xpoint_start=2, n_slots=2):
    """Matched-null position RMSE (m) + count accuracy for the X-point SET.

    For each TEST sample, the predicted slot means are matched to the PRESENT
    target nulls by the min-cost (Euclidean) assignment (K=2: min over the 2
    permutations), and the matched (R, Z) distances are pooled into an RMSE.
    Count accuracy compares the predicted count (presence_prob>=0.5 per sorted
    slot -> 0/1/2) to the true present-null count.  Returns a dict.
    """
    s = xpoint_start
    pr = [pred[:, s + 2 * k : s + 2 * k + 2] for k in range(n_slots)]
    tr = [y[:, s + 2 * k : s + 2 * k + 2] for k in range(n_slots)]
    present = [(mask[:, s + 2 * k]) & (mask[:, s + 2 * k + 1]) for k in range(n_slots)]
    n_samp = pred.shape[0]
    pred_count = (presence_prob >= 0.5).sum(axis=1)
    true_count = np.array(
        [sum(int(present[k][i]) for k in range(n_slots)) for i in range(n_samp)]
    )

    # per-null matched squared (R,Z) distances (metres^2) for an honest RMSE.
    def _sq(a, b):
        return float(np.sum((a - b) ** 2))

    null_sq = []
    for i in range(n_samp):
        tgt = [k for k in range(n_slots) if present[k][i]]
        if len(tgt) == 2:
            d_id = _sq(pr[0][i], tr[0][i]) + _sq(pr[1][i], tr[1][i])
            d_sw = _sq(pr[0][i], tr[1][i]) + _sq(pr[1][i], tr[0][i])
            if d_id <= d_sw:
                null_sq += [_sq(pr[0][i], tr[0][i]), _sq(pr[1][i], tr[1][i])]
            else:
                null_sq += [_sq(pr[0][i], tr[1][i]), _sq(pr[1][i], tr[0][i])]
        elif len(tgt) == 1:
            t = tgt[0]
            null_sq.append(min(_sq(pr[k][i], tr[t][i]) for k in range(n_slots)))
    matched_rmse = float(np.sqrt(np.mean(null_sq))) if null_sq else float("nan")
    count_acc = float(np.mean(pred_count == true_count)) if n_samp else float("nan")
    return {
        "matched_null_rmse_m": matched_rmse,
        "count_accuracy": count_acc,
        "n_matched_nulls": int(len(null_sq)),
        "true_count_dist": {int(c): int((true_count == c).sum()) for c in (0, 1, 2)},
    }


def per_component_rmse(pred, y, mask):
    """Per-component RMSE in metres over finite-label TEST elements."""
    dim = y.shape[1]
    out = np.full(dim, np.nan)
    for d in range(dim):
        sel = mask[:, d]
        if sel.sum() == 0:
            continue
        err = pred[sel, d] - y[sel, d]
        out[d] = float(np.sqrt(np.mean(err**2)))
    return out


def mean_predictor_rmse(ytr, mtr, yte, mte):
    """Baseline RMSE: predict the TRAIN mean for every TEST example."""
    dim = yte.shape[1]
    out = np.full(dim, np.nan)
    for d in range(dim):
        tr = ytr[mtr[:, d], d]
        te = yte[mte[:, d], d]
        if tr.size == 0 or te.size == 0:
            continue
        out[d] = float(np.sqrt(np.mean((te - float(np.mean(tr))) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def verdict(rmse_probe, rmse_baseline, names, *, ratio_threshold):
    """Feasibility verdict: probe RMSE materially below the mean-predictor.

    PASS: for axis + X-point components, probe RMSE < baseline / ratio_threshold.
    Reports per-component skill = 1 - probe/baseline.
    """
    rows = []
    # The headline PASS criterion is the magnetic AXIS (axis_R, axis_Z) — robust,
    # always present.  The split X-point channels (lower/upper) are reported
    # per-component but are present-when-present (often masked), so they do not
    # gate the overall feasibility flag; the reader judges them per-component.
    key = {"axis_R", "axis_Z"}
    axis_xpt_pass = []
    for d, nm in enumerate(names):
        rp = rmse_probe[d]
        rb = rmse_baseline[d]
        if rb is None or not np.isfinite(rb) or rb == 0 or not np.isfinite(rp):
            skill = None
            beats = None
        else:
            skill = 1.0 - rp / rb
            beats = bool(rp < rb / ratio_threshold)
        rows.append(
            {
                "component": nm,
                "rmse_probe_m": None if not np.isfinite(rp) else float(rp),
                "rmse_baseline_m": None
                if (rb is None or not np.isfinite(rb))
                else float(rb),
                "skill": None if skill is None else float(skill),
                "beats_baseline": beats,
            }
        )
        if nm in key and beats is not None:
            axis_xpt_pass.append(beats)
    overall = bool(axis_xpt_pass) and all(axis_xpt_pass)
    return {
        "feasible": overall,
        "criterion": (
            f"probe RMSE < baseline / {ratio_threshold:g} for axis_R + axis_Z "
            "(the robust always-present components); split lower/upper X-point "
            "channels reported per-component (present-when-present)"
        ),
        "ratio_threshold": ratio_threshold,
        "components": rows,
    }


# ---------------------------------------------------------------------------
# Scatter figure (axis + X-point)
# ---------------------------------------------------------------------------


def geometry_scatter(pred, y, mask, names, out_path, *, title):
    """Pred-vs-true scatter for axis + the split lower/upper X-point heights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {nm: d for d, nm in enumerate(names)}
    # Judge on axis + LCFS (X-point deferred); show axis + two LCFS control radii.
    wanted = ["axis_R", "axis_Z", "lcfs_r_2", "lcfs_r_6"]
    comps = [(idx[nm], nm) for nm in wanted if nm in idx]
    fig, axes = plt.subplots(
        1, len(comps), figsize=(4 * len(comps), 4.2), constrained_layout=True
    )
    if len(comps) == 1:
        axes = [axes]
    for ax, (d, label) in zip(axes, comps, strict=True):
        sel = mask[:, d]
        if sel.sum() == 0:
            ax.set_title(f"{label}: no finite test labels")
            continue
        yt = y[sel, d]
        yp = pred[sel, d]
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.scatter(yt, yp, s=14, alpha=0.55, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ax.set_xlabel(f"true {label} (m)")
        ax.set_ylabel(f"predicted {label} (m)")
        ax.set_title(f"{label}  RMSE={rmse * 100:.1f} cm  (n={int(sel.sum())})")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# One arm = (train probe on a stream subset, eval, metrics)
# ---------------------------------------------------------------------------


def run_arm(
    tr_examples,
    te_examples,
    channels,
    modalities,
    args,
    *,
    restrict,
    label,
    continuous=False,
):
    """Train + evaluate one ablation arm; returns (report_dict, pred, yte, mte).

    ``continuous`` selects the value-input lane: ``False`` (default) embeds the
    256-bin quantised token id; ``True`` embeds the CONTINUOUS corpus-standardised
    raw value (the quantisation ablation).
    """
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_DIM, TARGET_NAMES

    specs = build_stream_specs(channels, modalities, restrict=restrict)
    if not specs:
        logger.warning("arm '%s': no streams present — skipping", label)
        return None
    device = "cuda" if _cuda_available() else "cpu"
    model, tmean, tstd = train_probe(
        tr_examples,
        specs,
        n_steps=args.n_signal_steps,
        target_dim=TARGET_DIM,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        continuous=continuous,
    )
    pred, yte, mte, presence_prob = evaluate(
        model,
        te_examples,
        specs,
        tmean,
        tstd,
        n_steps=args.n_signal_steps,
        device=device,
        batch_size=args.batch_size,
        continuous=continuous,
    )
    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    rmse_probe = per_component_rmse(pred, yte, mte)
    rmse_base = mean_predictor_rmse(ytr, mtr, yte, mte)
    verd = verdict(
        rmse_probe, rmse_base, TARGET_NAMES, ratio_threshold=args.ratio_threshold
    )
    # X-point null SET metrics (matched-null RMSE + count accuracy) — the
    # order-invariant X-point verdict (NOT fixed per-slot components).
    xpt_set = set_xpoint_metrics(pred, yte, mte, presence_prob)
    report = {
        "arm": label,
        "value_input": "continuous" if continuous else "quantised_token",
        "streams": [s.name for s in specs],
        "stream_channels": {s.name: s.channels for s in specs},
        "probe_params_M": round(model.n_parameters() / 1e6, 3),
        "verdict": verd,
        "xpoint_set": xpt_set,
    }
    logger.info(
        "  X-POINT SET: matched-null RMSE=%.1fcm  count-acc=%.2f  (n_nulls=%d, "
        "true-count %s)",
        xpt_set["matched_null_rmse_m"] * 100
        if np.isfinite(xpt_set["matched_null_rmse_m"])
        else float("nan"),
        xpt_set["count_accuracy"],
        xpt_set["n_matched_nulls"],
        xpt_set["true_count_dist"],
    )
    # console summary
    logger.info(
        "=== ARM '%s' VERDICT: %s ===",
        label,
        "FEASIBLE" if verd["feasible"] else "INFEASIBLE",
    )
    for row in verd["components"]:
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
    return report, pred, yte, mte


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args) -> int:
    from imas_ambix.camdyn.dataset import list_token_shot_ids
    from imas_ambix.camdyn.splits import build_camdyn_split
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_NAMES
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig
    from imas_ambix.worldmodel.spacetime_dataset_v2 import extended_signal_modalities

    rng = np.random.default_rng(args.seed)
    token_root = Path(args.token_root) if args.token_root else None
    level2_root = Path(args.level2_root) if args.level2_root else None
    modalities = extended_signal_modalities()

    # Absolute mode: load the corpus calibration for the STAGED streams (the
    # read-time-quantised groups, e.g. magnetics) so a given physical reading
    # maps to the same token in every shot.  The window / signal_hf streams
    # (xma, L2 light-path) instead carry absolute magnitude in their re-encoded
    # stores; at read time only the staged groups consult this dict.
    calibration_by_group = None
    if args.absolute:
        from imas_ambix.calibration.corpus_compute import load_group_calibration

        calibration_by_group = {}
        for m in modalities:
            if m.kind not in ("staged", "ait"):
                continue
            cal = load_group_calibration(m.group)
            if cal:
                calibration_by_group[m.group] = cal
                logger.info(
                    "absolute: loaded calibration for staged group %r (%d channels)",
                    m.group,
                    len(cal),
                )
            else:
                logger.warning(
                    "absolute: NO calibration for staged group %r — per-shot fallback",
                    m.group,
                )
        # PF-active coil currents are read directly from L2 (not a modality);
        # load their corpus calibration too so the coil-current values are
        # corpus-standardised on the SAME absolute scale as the magnetics.
        pf_cal = load_group_calibration("pf_active")
        if pf_cal:
            calibration_by_group["pf_active"] = pf_cal
            logger.info(
                "absolute: loaded calibration for pf_active (%d channels)",
                len(pf_cal),
            )
        else:
            logger.warning(
                "absolute: NO calibration for pf_active — coil currents per-shot"
            )
        if not calibration_by_group:
            logger.warning(
                "absolute requested but no staged calibration on disk — "
                "staged streams stay per-shot (run corpus_compute first)"
            )

    config = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=8,
        context_frames=max(1, args.n_frames // 3),
        target_horizon_s=args.target_horizon_s,
    )

    # 1) split — force the gate cohort + held-out into the oracle TEST set.
    all_shots = list_token_shot_ids(camera=args.camera, token_root=token_root)
    logger.info("rbb token shots on disk: %d", len(all_shots))
    split = build_camdyn_split(
        all_shots,
        mse_heldout=list(FORCED_TEST_SHOTS),
        val_fraction=0.0,
        held_out_fraction=args.held_out_fraction,
        seed=args.seed,
    )
    train_pool = list(split.train)
    test_pool = list(split.held_out)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    train_shots = train_pool[: args.n_train_shots]
    test_shots = test_pool[: args.n_test_shots]
    forced_present = [s for s in FORCED_TEST_SHOTS if s in set(all_shots)]
    test_shots = sorted(set(test_shots) | set(forced_present))
    logger.info(
        "TRAIN shots=%d  TEST shots=%d (forced present: %s)",
        len(train_shots),
        len(test_shots),
        forced_present,
    )

    # 2-4) assemble examples (window span -> measured signals -> 12-D labels)
    tr_examples = assemble_examples(
        train_shots,
        camera=args.camera,
        modalities=modalities,
        n_signal_steps=args.n_signal_steps,
        config=config,
        level2_root=level2_root,
        token_root=token_root,
        calibration_by_group=calibration_by_group,
    )
    te_examples = assemble_examples(
        test_shots,
        camera=args.camera,
        modalities=modalities,
        n_signal_steps=args.n_signal_steps,
        config=config,
        level2_root=level2_root,
        token_root=token_root,
        calibration_by_group=calibration_by_group,
    )
    if not tr_examples or not te_examples:
        logger.error("empty TRAIN or TEST example set — cannot run oracle")
        return 2
    logger.info(
        "TRAIN examples=%d  TEST examples=%d", len(tr_examples), len(te_examples)
    )

    channels = probe_channels(tr_examples + te_examples, modalities)
    logger.info("probed channels: %s", {k: v for k, v in channels.items() if v > 0})

    # 5-6) ALL-SIGNALS CONSOLIDATION (eval path).  Both arms CONTINUOUS, same
    # cohort/seed, one run, every stream routed through the SHARED space-time
    # relational encoder (per-(sensor,time) tokens + periodic geometry PE + STFT,
    # NO pooling) with geometry for ALL channels:
    #   arm A: MAGNETICS-ONLY baseline (xma + the staged magnetics array).
    #   arm B: ALL DIAGNOSTICS consolidated (restrict=None) — magnetics + the
    #          toroidal saddle + coils + soft_x_rays (chord geom) + interferometer
    #          + summary + gas_injection + xsx/xim + ait + ada/adg/aim, every
    #          channel geometry-tagged + continuous.
    # B − A on axis_R/axis_Z/LCFS = whether the extra streams now CONTRIBUTE
    # through the shared geometry-aware encoder (the all-diagnostics arm
    # previously overfit / did not beat magnetics-only without geometry).
    magnetics_only = {"xma", "magnetics"}

    arm_a = run_arm(
        tr_examples,
        te_examples,
        channels,
        modalities,
        args,
        restrict=magnetics_only,
        label="magnetics_only_continuous",
        continuous=True,
    )
    arm_b = run_arm(
        tr_examples,
        te_examples,
        channels,
        modalities,
        args,
        restrict=None,
        label="all_diagnostics_consolidated",
        continuous=True,
    )
    if arm_a is None or arm_b is None:
        logger.error("consolidation arm A or B produced no streams — abort")
        return 3

    # the headline result the report + scatter mirror is arm B (all-diagnostics).
    all_report, all_pred, all_yte, all_mte = arm_b

    # log the consolidation delta B − A on axis + LCFS (the open question:
    # does the geometry-tagged all-diagnostics arm beat magnetics-only?).
    def _skill(arm, comp):
        for row in arm[0]["verdict"]["components"]:
            if row["component"] == comp:
                return row["skill"]
        return None

    for comp in ("axis_R", "axis_Z", "lcfs_r_0", "lcfs_r_2", "lcfs_r_4", "lcfs_r_6"):
        sa, sb = _skill(arm_a, comp), _skill(arm_b, comp)
        if sa is not None and sb is not None:
            logger.info(
                "CONSOLIDATION  %-10s: A(magnetics)=%+.3f  "
                "B(all-diag)=%+.3f  delta=%+.3f",
                comp,
                sa,
                sb,
                sb - sa,
            )

    coverage = {
        "train_examples": len(tr_examples),
        "test_examples": len(te_examples),
        "train_shots": [int(s) for s in train_shots],
        "test_shots": [int(s) for s in test_shots],
        "forced_test_present": [int(s) for s in forced_present],
        "probed_channels": {k: int(v) for k, v in channels.items() if v > 0},
        "n_signal_steps": args.n_signal_steps,
        "target_horizon_s": args.target_horizon_s,
        "calibration_mode": "absolute" if args.absolute else "per_shot",
        "calibrated_staged_groups": sorted(calibration_by_group)
        if calibration_by_group
        else [],
    }
    report = {
        "task": "diagnostics feasibility oracle (measured signals -> geometry)",
        "evaluator_only": True,
        "camera": args.camera,
        "n_frames": args.n_frames,
        "n_signal_steps": args.n_signal_steps,
        "epochs": args.epochs,
        "target_names": list(TARGET_NAMES),
        "target_units": "m",
        "coverage": coverage,
        "arms": {
            # all-signals consolidation (eval path; both continuous, same cohort):
            "magnetics_only_continuous": (arm_a[0] if arm_a is not None else None),
            # HEADLINE: ALL diagnostics consolidated through the shared encoder
            # (B − A = whether the extra geometry-tagged streams contribute).
            "all_diagnostics_consolidated": all_report,
        },
        # top-level verdict mirrors the all-diagnostics consolidated arm.  Judged
        # on axis + LCFS (the X-point is DEFERRED — its single-primary target is
        # bimodal across topology switches; not a verdict here).
        "verdict": all_report["verdict"],
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "oracle_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", json_path)

    # scatter for the headline all-diagnostics consolidated arm.
    title = (
        "diagnostics feasibility oracle — ALL diagnostics consolidated "
        "(shared geometry-aware encoder, continuous) -> plasma geometry"
    )
    fig_local = out_root / "fig-diag-eq-oracle-geometry-scatter.png"
    geometry_scatter(all_pred, all_yte, all_mte, TARGET_NAMES, fig_local, title=title)
    fig_docs = Path(args.fig_dir) / "fig-diag-eq-oracle-geometry-scatter.png"
    try:
        geometry_scatter(
            all_pred, all_yte, all_mte, TARGET_NAMES, fig_docs, title=title
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write docs figure %s: %s", fig_docs, exc)

    logger.info(
        "=== TOP-LEVEL (efit-inputs continuous) FEASIBILITY: %s ===",
        "FEASIBLE" if report["verdict"]["feasible"] else "INFEASIBLE",
    )
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="rbb")
    p.add_argument("--n-frames", type=int, default=24, help="camera frames per window")
    p.add_argument(
        "--target-horizon-s",
        type=float,
        default=0.25,
        help="physical time span a window covers (s)",
    )
    p.add_argument(
        "--n-signal-steps",
        type=int,
        default=12,
        help="measured-signal temporal positions across the window span",
    )
    p.add_argument("--n-train-shots", type=int, default=250)
    p.add_argument("--n-test-shots", type=int, default=60)
    p.add_argument("--held-out-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.3,
        help="probe must beat baseline by this factor on axis+X-point",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-root", default=None, help="override token root")
    p.add_argument("--level2-root", default=None, help="override L2 equilibrium root")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    p.add_argument(
        "--absolute",
        action="store_true",
        help="standardise staged streams against the persisted CORPUS calibration "
        "(absolute magnitude survives tokenisation) instead of per-shot",
    )
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
