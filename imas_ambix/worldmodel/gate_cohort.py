"""Screen a TRAIN-DISJOINT eval-only cohort for the controllability gate.

The fixed 18502-05 held-out set is effectively 1-shot (frame evidence): only
18502 is a usable controllability probe.  18503/18504 are degenerate-by-metric
or degenerate-by-data — their ground-truth forecast window is dim or near-static,
so the model collapses to the same dream regardless of plan and the gate
FALSELY fails.  This module builds a larger, SCREENED cohort instead, with no
re-train: it enumerates candidate shots that are disjoint from the training
manifest and have a usable rbb recording, then keeps only the shots whose
GROUND-TRUTH forecast window is bright enough and moves enough for the gate to
be a fair test.

A shot PASSES the screen when, over its assembled forecast window, the decoded
ground-truth:

* mean brightness >= ``min_brightness`` (kills dark shots like 18504, which the
  model dreams black regardless of plan -> a degenerate true-vs-random of 0);
* frame-to-frame transient motion (mean pixel-L1 between consecutive forecast
  frames) >= ``min_transient_motion`` (the plasma must actually MOVE — a static
  forecast has no dynamics for the plan to steer);
* the in-window actuator PLAN varies (reuses :func:`find_transient_window`'s
  variation score; a flat-top window has no control variation to respond to).

The cohort + per-shot screen stats are written to JSON so the eval reads it via
``--held-out-cohort PATH``.  A binding LEAKAGE GUARD asserts that ZERO cohort
shots appear in the training manifest.

GPU-safety (AGENTS.md §2b): the GT decode loads the frozen VQ **exactly once**
for the whole cohort.  Every assemblable candidate's ``(n_frames, 16, 16)`` GT
store grid is stacked into one ``(N, n_frames, 16, 16)`` batch and decoded inside
a SINGLE Open-MAGVIT2-venv subprocess that loads the VQModel once, decodes in
memory-bounded internal batches, and returns only the per-frame GRAY luminance
stats (never the multi-GB RGB stacks) — so there is no per-shot/per-chunk model
reload (the prior builder reloaded the model once per 8-candidate chunk → ~28
reloads for a few-hundred-shot scan) and no giant image bundle on the FS.  The
candidate enumeration + train-disjoint set + leakage guard + screen scoring are
pure-numpy / filesystem and CPU-testable.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: The Open-MAGVIT2 venv interpreter that owns the frozen VQ decoder weights (the
#: ambix venv has no torch/omegaconf for it; the decode therefore runs as a
#: subprocess under this interpreter — mirrors ``reconstruction_demo``).
MAGVIT2_PYTHON = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python"
)
MAGVIT2_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/open-magvit2")
#: Registry shift between stored (global) token ids and the local LFQ codebook ids
#: the VQ decoder expects (``len(CONTROL_TOKENS) == 4``).
REGISTRY_OFFSET = 4
GRID_H, GRID_W = 16, 16

#: Default location the screened cohort is written to / read from.
DEFAULT_COHORT_PATH = Path("/work/projects/imas_gpu/worldmodel/gate_cohort.json")

#: Default token store root (rbb frames + signal stores live under ``<root>/v1``).
DEFAULT_TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")


# ---------------------------------------------------------------------------
# Screen thresholds + per-shot record
# ---------------------------------------------------------------------------


@dataclass
class ScreenThresholds:
    """Cohort-screen gates (see module docstring for the rationale)."""

    #: Min GT forecast-window mean brightness (0-255 luminance).  18504 fails at
    #: 2.7; 18502 passes at 17.
    min_brightness: float = 10.0
    #: Min GT forecast-window frame-to-frame pixel-L1 (transient motion).  18504
    #: fails at 2.3, 18503 is borderline at 3.9; 18502 passes at 8.2.
    min_transient_motion: float = 4.0
    #: Min in-window actuator-plan variation (summed per-channel std) for the
    #: window to be a fair controllability probe.
    min_plan_variation: float = 1e-3
    #: Min number of measured-signal streams that actually load for the shot, so
    #: the eval conditions on a real multi-modal context (not camera-only).
    min_streams: int = 3
    #: When too few candidates pass at the default brightness/motion gates, the
    #: builder may RELAX brightness + motion down to these floors (in steps) to
    #: reach ``target_size`` — never relaxing plan-variation or stream-count (those
    #: are correctness gates, not just "is this shot a fair probe").  The relaxed
    #: values actually used are recorded in the cohort JSON.
    min_brightness_floor: float = 7.0
    min_transient_motion_floor: float = 3.0

    def relax_step(self) -> ScreenThresholds:
        """A modestly-relaxed copy (brightness/motion ↓ toward their floors).

        Halves the gap to the floor each call (geometric approach), so a few
        calls reach the floor without an abrupt jump.  Plan-variation and
        stream-count are correctness gates and never relaxed.
        """
        return ScreenThresholds(
            min_brightness=max(
                self.min_brightness_floor,
                self.min_brightness_floor
                + 0.5 * (self.min_brightness - self.min_brightness_floor),
            ),
            min_transient_motion=max(
                self.min_transient_motion_floor,
                self.min_transient_motion_floor
                + 0.5 * (self.min_transient_motion - self.min_transient_motion_floor),
            ),
            min_plan_variation=self.min_plan_variation,
            min_streams=self.min_streams,
            min_brightness_floor=self.min_brightness_floor,
            min_transient_motion_floor=self.min_transient_motion_floor,
        )

    def at_floor(self) -> bool:
        """True once brightness AND motion have reached their relaxation floors."""
        return (
            self.min_brightness <= self.min_brightness_floor + 1e-9
            and self.min_transient_motion <= self.min_transient_motion_floor + 1e-9
        )


@dataclass
class DriveabilityThresholds:
    """Eligibility gates for the DRIVEABLE-enriched cohort (model-free).

    The driveable cohort is built on top of the same brightness / stream / span
    eligibility gates as :class:`ScreenThresholds` (a candidate must still be a
    bright, multi-modal, horizon-spanning window), but instead of a fixed
    brightness/motion screen the passers are RANKED by a physical driveability
    score (:func:`driveability_score`) and the top ``target_size`` train-disjoint
    shots are kept.

    The thresholds below are ELIGIBILITY floors — a candidate must clear every
    one to be eligible for ranking; they are deliberately permissive so the
    *ranking* (not a knife-edge cut) does the enrichment.

    Anti-circularity (binding): every quantity scored here is a property of the
    GROUND-TRUTH data + the DEMANDED commands ONLY — never of any world-model's
    response.  ``command_*`` come from the demanded actuator plan; ``centroid_*``
    and ``motion`` come from the decoded GT frames.  No model rollout is consulted.
    """

    #: Min GT forecast mean brightness — kills dark shots the decoder dreams black
    #: regardless of plan (same floor as the screen's relaxation floor).
    min_brightness: float = 7.0
    #: Min number of measured-signal streams (the eval conditions on a real
    #: multi-modal context, not camera-only).
    min_streams: int = 3
    #: Min demanded-command variation over the window (a flat-top shot has nothing
    #: to control — there is no actuation to respond to).
    min_plan_variation: float = 1e-3
    #: Min GT forecast transient motion (the plasma must be non-static for there
    #: to be any drivable dynamics at all).
    min_transient_motion: float = 2.0
    #: Min GT centroid path length (px) — the emission centroid must actually
    #: TRAVEL over the window (a brightening-in-place blob is not driveable motion).
    min_centroid_path_px: float = 1.0

    def eligibility_fails(self, rec: CohortShotScreen) -> list[str]:
        """The list of eligibility-gate failures for *rec* (empty == eligible)."""
        fails: list[str] = []
        if rec.mean_brightness < self.min_brightness:
            fails.append(f"brightness {rec.mean_brightness:.1f}<{self.min_brightness}")
        if rec.n_streams < self.min_streams:
            fails.append(f"streams {rec.n_streams}<{self.min_streams}")
        if rec.plan_variation < self.min_plan_variation:
            fails.append(f"plan_var {rec.plan_variation:.2e}<{self.min_plan_variation}")
        if rec.transient_motion < self.min_transient_motion:
            fails.append(
                f"motion {rec.transient_motion:.1f}<{self.min_transient_motion}"
            )
        if rec.centroid_path_px < self.min_centroid_path_px:
            fails.append(
                f"centroid_path {rec.centroid_path_px:.1f}<{self.min_centroid_path_px}"
            )
        return fails


@dataclass
class CohortShotScreen:
    """The screen outcome + GT stats for one candidate shot."""

    shot_id: int
    assemblable: bool
    passed: bool
    reason: str = ""
    recording_span_s: float = 0.0
    achieved_window_span_s: float = 0.0
    context_frames: int = 0
    n_frames: int = 0
    mean_brightness: float = 0.0
    p99_brightness: float = 0.0
    transient_motion: float = 0.0
    plan_variation: float = 0.0
    n_streams: int = 0
    stream_names: list = field(default_factory=list)

    # --- driveability components (model-free; only filled in `driveable` mode) ---
    #: Total path length of the GT emission centroid over the forecast window
    #: (sum of consecutive-frame centroid displacements, in pixels).  A plasma
    #: that physically MOVES has a long path; a static blob has ~0.
    centroid_path_px: float = 0.0
    #: Per-frame command change (L1 of the demanded actuator vector between
    #: consecutive plan steps), summed across present channels.  How hard the
    #: commands push the plasma at each step.
    command_change_total: float = 0.0
    #: Temporal-association proxy in [0, 1]: do the demanded commands change at
    #: the SAME time the GT centroid moves?  Cosine overlap of the (interpolated)
    #: per-frame command-change profile and the per-frame centroid-displacement
    #: profile over the forecast window — a lightweight, model-free correlation.
    temporal_association: float = 0.0
    #: Combined PHYSICAL driveability score (model-free) used to RANK candidates
    #: in `driveable` mode.  See :func:`driveability_score`.
    driveability_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidate enumeration (filesystem + train-disjoint)
# ---------------------------------------------------------------------------


def training_shot_ids(manifest_path: str | Path) -> set[int]:
    """The set of shot ids used by the TRAINING manifest (cohort must avoid these).

    Reads the curated windows manifest (``{..., "windows": [{"shot_id": ...},
    ...]}``) and returns every distinct ``shot_id`` it draws windows from.  Also
    folds in any ids the manifest declares as ``held_out`` (the legacy fixed set)
    so the new cohort never re-uses them either.
    """
    data = json.loads(Path(manifest_path).read_text())
    ids: set[int] = set()
    windows = data.get("windows") or []
    for w in windows:
        if isinstance(w, dict) and "shot_id" in w:
            ids.add(int(w["shot_id"]))
        elif isinstance(w, (list, tuple)) and w:
            ids.add(int(w[0]))  # schema[0] == shot_id
    for s in data.get("held_out") or []:
        ids.add(int(s))
    return ids


def assert_disjoint(
    cohort_ids, train_ids, *, manifest_label: str = "training manifest"
):
    """Binding leakage guard: NO cohort shot may appear in the training set.

    Raises :class:`AssertionError` listing the offending ids if any cohort shot
    is also a training shot.  Called by :func:`build_screened_cohort` before the
    cohort is written and unit-tested on a synthetic manifest + candidate set.
    """
    leak = sorted(set(int(s) for s in cohort_ids) & set(int(s) for s in train_ids))
    assert not leak, (
        f"cohort leakage: {len(leak)} shot(s) appear in the {manifest_label} "
        f"and the eval cohort — {leak}"
    )


def enumerate_candidate_shots(
    *,
    token_root: Path,
    camera: str,
    train_ids: set[int],
    cap: int = 60,
    require_magnetics: bool = True,
) -> list[int]:
    """Train-disjoint shot ids that have a camera recording (+ optional magnetics).

    Scans ``<token_root>/v1/frames/<shot>/<camera>.zarr`` for shots with a
    recording, drops any in ``train_ids`` (the leakage guard), optionally requires
    the magnetics signal store (the direct plasma-position sensor — the most
    valuable conditioning stream), and returns up to ``cap`` ids sorted ascending
    (deterministic).  The per-shot brightness / motion screen is applied later by
    :func:`screen_candidate` once the window is assembled + decoded.
    """
    frames_root = Path(token_root) / "v1" / "frames"
    mag_root = Path(token_root) / "v1" / "signals-magnetics" / "magnetics"
    if not frames_root.is_dir():
        raise FileNotFoundError(f"no frames root at {frames_root}")
    out: list[int] = []
    for child in sorted(frames_root.iterdir(), key=lambda p: _shot_sort_key(p.name)):
        if len(out) >= cap:
            break
        name = child.name
        if not name.isdigit():
            continue
        sid = int(name)
        if sid in train_ids:
            continue
        if not (child / f"{camera}.zarr").is_dir():
            continue
        if require_magnetics and not (mag_root / name / "magnetics.zarr").is_dir():
            continue
        out.append(sid)
    return out


def _shot_sort_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


# ---------------------------------------------------------------------------
# Per-shot screen (assemble window, decode GT, score brightness + motion)
# ---------------------------------------------------------------------------


def _assemble_for_screen(shot_id, cfg, *, camera, token_root):
    """Assemble one candidate window (CPU only — no decode).

    Returns ``(record, store_frames, ctx)`` where ``record`` is a partly-filled
    :class:`CohortShotScreen` carrying the cheap stats (span, plan variation,
    stream count); ``store_frames`` is the ``(F,16,16)`` GT store-id grid to decode
    later (``None`` when unassemblable).  Splitting assembly from decode lets the
    builder BATCH all GT decodes in a few VQ passes (model loaded once) instead of
    a subprocess per shot.
    """
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _assemble_heldout,
        _plan_variation,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        recording_time_span_s,
    )

    span_s = recording_time_span_s(int(shot_id), camera=camera, token_root=token_root)
    try:
        sample = _assemble_heldout(
            int(shot_id), cfg, camera=camera, token_root=token_root
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return (
            CohortShotScreen(
                shot_id=int(shot_id),
                assemblable=False,
                passed=False,
                reason=f"unassemblable: {exc!r}",
                recording_span_s=float(span_s or 0.0),
            ),
            None,
            0,
        )

    base = sample.signal.base
    ctx = int(base.context_frames)
    ft = np.asarray(base.frame_time, dtype=np.float64)
    stream_names = list(sample.signals.keys())
    rec = CohortShotScreen(
        shot_id=int(shot_id),
        assemblable=True,
        passed=False,
        recording_span_s=float(span_s or 0.0),
        achieved_window_span_s=float(ft[-1] - ft[0]) if ft.size >= 2 else 0.0,
        context_frames=ctx,
        n_frames=int(base.n_frames),
        plan_variation=float(_plan_variation(sample)),
        n_streams=len(stream_names),
        stream_names=stream_names,
    )
    return rec, base.store_frames(), ctx


def _gt_window_stats(gt_gray, context_frames: int) -> tuple[float, float, float]:
    """``(mean_brightness, p99_brightness, transient_motion)`` over the forecast win.

    Pure-numpy on a decoded ``(F, H, W)`` gray stack — the CPU-testable core of the
    brightness/motion screen (no VQ decode needed to exercise it).
    """
    fwin = gt_gray[context_frames:]
    mean_bri = float(fwin.mean()) if fwin.size else 0.0
    p99_bri = float(np.percentile(fwin, 99)) if fwin.size else 0.0
    motion = float(np.abs(np.diff(fwin, axis=0)).mean()) if fwin.shape[0] >= 2 else 0.0
    return mean_bri, p99_bri, motion


def _score_gt(rec, gt_gray, thresholds):
    """Fill the GT brightness/motion stats + pass/fail on a decoded GT stack.

    CPU-testable: hand it a synthetic ``(F, H, W)`` gray stack and a record with
    ``context_frames`` set and it computes the stats + applies the screen — the
    same gate the GPU builder applies to its decoded candidates.
    """
    mean_bri, p99_bri, motion = _gt_window_stats(gt_gray, rec.context_frames)
    return _apply_gt_stats(rec, mean_bri, p99_bri, motion, thresholds)


# ---------------------------------------------------------------------------
# Physical, MODEL-FREE driveability scoring (GT + demanded commands ONLY)
# ---------------------------------------------------------------------------
#
# Anti-circularity is the whole point: a candidate's driveability is measured
# from (a) the demanded actuator plan (does the operator ASK the plasma to
# move?) and (b) the decoded GROUND-TRUTH frames (does the plasma physically
# RESPOND — emission moving and centroid travelling?).  A world-model's own
# rollout is NEVER consulted, so the screen cannot select shots the model
# already drives and inflate the gate.


def _centroid_displacement_profile(gt_gray, context_frames: int) -> np.ndarray:
    """Per-step GT centroid displacement (px) over the forecast window.

    Reuses :func:`controllable_eval.decoded_centroid` (the SAME emission-centroid
    the gate uses) on the decoded GROUND-TRUTH gray stack, then returns the
    consecutive-frame centroid step lengths over the forecast frames — a
    ``(n_forecast - 1,)`` profile of how far the plasma blob travels each step.
    Empty when the forecast window has < 2 frames.
    """
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        decoded_centroid,
    )

    fwin = np.asarray(gt_gray, dtype=np.float64)[context_frames:]
    if fwin.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    cen = decoded_centroid(fwin)  # (F, 2) row/col
    return np.linalg.norm(np.diff(cen, axis=0), axis=1)


def _centroid_path_length(gt_gray, context_frames: int) -> float:
    """Total GT emission-centroid path length (px) over the forecast window.

    The sum of the per-step centroid displacements — how far the plasma blob
    travels in total.  A genuinely moving plasma has a long path; a blob that
    merely brightens in place has ~0.  Model-free (GT frames only).
    """
    return float(_centroid_displacement_profile(gt_gray, context_frames).sum())


def _command_change_profile(plan_values, plan_missing) -> np.ndarray:
    """Per-step demanded-command change (summed over present channels).

    ``plan_values`` is the ``(n_plan, C)`` normalised demanded actuator surface
    and ``plan_missing`` its ``(n_plan, C)`` missing-flag.  Returns a
    ``(n_plan - 1,)`` profile: the L1 of the demanded vector between consecutive
    plan steps, summed over the channels that are present for the window.  This is
    a property of the DEMANDED commands only — no model, no GT.
    """
    vals = np.asarray(plan_values, dtype=np.float64)
    miss = np.asarray(plan_missing, dtype=np.float64)
    if vals.ndim != 2 or vals.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    present = miss.mean(axis=0) < 1.0 if miss.size else np.ones(vals.shape[1], bool)
    if not bool(np.asarray(present).any()):
        return np.zeros((vals.shape[0] - 1,), dtype=np.float64)
    return np.abs(np.diff(vals[:, present], axis=0)).sum(axis=1)


def _resample_profile(profile: np.ndarray, n: int) -> np.ndarray:
    """Linear-resample a 1-D profile onto ``n`` evenly-spaced samples.

    Used to put the command-change profile (one sample per plan step) and the
    centroid-displacement profile (one sample per forecast frame) on a COMMON
    axis so they can be correlated.  Returns zeros when either side is empty.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if n <= 0 or profile.size == 0:
        return np.zeros((max(n, 0),), dtype=np.float64)
    if profile.size == 1:
        return np.full((n,), float(profile[0]), dtype=np.float64)
    src = np.linspace(0.0, 1.0, profile.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, profile)


def _temporal_association(command_profile, centroid_profile) -> float:
    """Model-free overlap in [0, 1] of WHEN commands change vs WHEN the plasma moves.

    Resamples both per-step profiles onto a common axis and returns their cosine
    similarity (clamped to ``[0, 1]``).  Near 1 when the demanded commands change
    at the same time the GT centroid travels (the response is plausibly CAUSED by
    the command); near 0 when command changes and plasma motion are temporally
    decoupled.  A lightweight, documented correlation — NOT a model rollout.
    """
    cmd = np.asarray(command_profile, dtype=np.float64)
    cen = np.asarray(centroid_profile, dtype=np.float64)
    if cmd.size == 0 or cen.size == 0:
        return 0.0
    n = max(cmd.size, cen.size)
    a = _resample_profile(cmd, n)
    b = _resample_profile(cen, n)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, cos))


def driveability_score(
    *,
    plan_variation: float,
    transient_motion: float,
    centroid_path_px: float,
    temporal_association: float,
) -> float:
    """Combined PHYSICAL driveability score (model-free) used to RANK candidates.

    Higher == more driveable: the demanded commands move (``plan_variation``), the
    GT plasma responds transiently (``transient_motion``) AND its emission centroid
    travels (``centroid_path_px``), and the command changes overlap in time with
    the plasma motion (``temporal_association`` proxy).  All four are properties of
    the GROUND-TRUTH data + the demanded commands ONLY — never of a world-model's
    response — so ranking by this score cannot select shots the model already
    drives (the anti-circularity constraint).

    The score multiplies a saturating transform of each component so a candidate
    must be strong on ALL of {commands move, plasma moves, centroid travels} —
    a flat-command shot (no actuation) or a static-plasma shot (no response)
    scores ~0 even if the other components are large.  The temporal-association
    proxy is a multiplicative bonus in ``[0.5, 1.0]`` (never zeroing the score,
    since it is the weakest/most heuristic signal) that up-weights candidates
    whose command changes and plasma motion coincide.
    """

    def _sat(x: float, scale: float) -> float:
        # bounded, monotone, ~0 at 0 and -> 1 for x >> scale.
        x = max(0.0, float(x))
        return x / (x + float(scale))

    cmd = _sat(plan_variation, 1.0)
    mot = _sat(transient_motion, 4.0)
    cen = _sat(centroid_path_px, 4.0)
    assoc = 0.5 + 0.5 * max(0.0, min(1.0, float(temporal_association)))
    return float(cmd * mot * cen * assoc)


def _apply_driveability(rec, *, command_profile=None, centroid_profile=None):
    """Fill *rec*'s driveability components + combined score (model-free).

    ``rec`` already carries ``plan_variation`` + ``transient_motion`` (from the
    CPU assemble) and ``centroid_path_px`` (from the GT decode).  ``*_profile`` are
    the optional per-step profiles used for the temporal-association proxy; when
    omitted (e.g. unit tests scoring cached scalars) association defaults to 0.
    Returns *rec* for chaining.
    """
    if command_profile is not None and centroid_profile is not None:
        rec.command_change_total = float(np.asarray(command_profile).sum())
        rec.temporal_association = _temporal_association(
            command_profile, centroid_profile
        )
    rec.driveability_score = driveability_score(
        plan_variation=rec.plan_variation,
        transient_motion=rec.transient_motion,
        centroid_path_px=rec.centroid_path_px,
        temporal_association=rec.temporal_association,
    )
    return rec


def _select_driveable_cohort(records, thresholds, target_size):
    """Keep the top-``target_size`` ELIGIBLE records by driveability score.

    Filters to the records that clear every :class:`DriveabilityThresholds`
    eligibility floor, RANKS them by ``driveability_score`` descending (shot id
    ascending as a deterministic tie-break), and returns the top ``target_size``
    shot ids.  No re-decode — scores on the cached per-record stats.
    """
    eligible = [
        rec
        for rec in records
        if rec.assemblable and not thresholds.eligibility_fails(rec)
    ]
    eligible.sort(key=lambda r: (-float(r.driveability_score), int(r.shot_id)))
    return [int(r.shot_id) for r in eligible[: int(target_size)]]


# ---------------------------------------------------------------------------
# In-process GT decode (model loaded ONCE for the whole cohort — AGENTS.md §2b)
# ---------------------------------------------------------------------------


def _screen_decode_phase(
    token_bundle: str, stats_bundle: str, device: str, decode_batch_size: int
) -> None:
    """Decode every candidate GT grid, emit per-shot GRAY stats (runs in venv).

    Loads the frozen VQModel ONCE, decodes the stacked ``(N, F, 16, 16)`` store-id
    grids in memory-bounded internal batches (so a few-hundred-shot scan never
    materialises a multi-GB RGB stack), and writes a small ``(N,)`` stats array:
    per-shot forecast-window mean / p99 brightness + frame-to-frame transient
    motion.  Invoked under ``MAGVIT2_PYTHON`` via :func:`_run_screen_decode`.

    This is the §2b in-process pattern realised inside the decoder venv: model
    load is outside the per-shot loop, decode happens in bounded batches, and the
    model + CUDA cache are released in ``finally``.
    """
    sys.path.insert(0, str(MAGVIT2_ROOT))
    from imas_ambix.bench.stream_worker import decode_batch, load_model  # noqa: PLC0415

    data = np.load(str(token_bundle), allow_pickle=True)
    grids = np.asarray(data["grids"], dtype=np.int64)  # (N, F, 16, 16) STORE ids
    ctx = np.asarray(data["context_frames"], dtype=np.int64)  # (N,) per-shot context
    n, f, h, w = grids.shape
    # store id -> local LFQ id (clamp the rare out-of-range / never-observed cell).
    local_all = np.clip(grids - REGISTRY_OFFSET, 0, (1 << 18) - 1).astype(np.int64)

    mean_bri = np.zeros(n, dtype=np.float64)
    p99_bri = np.zeros(n, dtype=np.float64)
    motion = np.zeros(n, dtype=np.float64)
    #: total GT emission-centroid path length over each shot's forecast window
    #: (model-free driveability component; see :func:`_centroid_path_length`).
    centroid_path = np.zeros(n, dtype=np.float64)
    #: per-shot, padded per-step centroid-displacement profiles (for the
    #: temporal-association proxy).  Fixed width = max forecast steps = F-1.
    cen_prof = np.zeros((n, max(f - 1, 1)), dtype=np.float64)

    model = load_model(MAGVIT2_ROOT, device)
    try:
        for i in range(n):
            # decode one shot's F frames (F is small, 24); decode_batch chunks
            # the F frames internally at decode_batch_size.
            rgb = decode_batch(
                model,
                local_all[i],
                device,
                max(1, int(decode_batch_size)),
                (h * 16, w * 16),
            )  # (F, 256, 256, 3) uint8
            gray = np.asarray(rgb, dtype=np.float64).mean(axis=-1)  # (F, 256, 256)
            c = int(ctx[i])
            fwin = gray[c:]
            if fwin.size:
                mean_bri[i] = float(fwin.mean())
                p99_bri[i] = float(np.percentile(fwin, 99))
            if fwin.shape[0] >= 2:
                motion[i] = float(np.abs(np.diff(fwin, axis=0)).mean())
            prof = _centroid_displacement_profile(gray, c)
            centroid_path[i] = float(prof.sum())
            if prof.size:
                cen_prof[i, : prof.size] = prof
    finally:
        try:
            del model
            if device.startswith("cuda"):
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    np.savez_compressed(
        str(stats_bundle),
        mean_brightness=mean_bri,
        p99_brightness=p99_bri,
        motion=motion,
        centroid_path=centroid_path,
        centroid_profile=cen_prof,
    )


def _run_screen_decode(
    grids: np.ndarray,
    context_frames: np.ndarray,
    *,
    work_dir: Path,
    device: str,
    decode_batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Decode the stacked candidate grids in ONE venv subprocess → per-shot stats.

    Writes the ``(N, F, 16, 16)`` store-id stack + per-shot context to a bundle,
    re-invokes this module under :data:`MAGVIT2_PYTHON` to load the VQ once and
    decode all candidates, and reads back the per-shot gray stats.  One model
    load for the whole cohort.
    """
    if not MAGVIT2_PYTHON.exists():
        raise RuntimeError(
            f"Open-MAGVIT2 decode interpreter not found at {MAGVIT2_PYTHON}. "
            "Cannot decode GT tokens to screen the cohort (no download on betelgeuse)."
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "screen_tokens.npz"
    stats_bundle = work_dir / "screen_stats.npz"
    np.savez_compressed(
        token_bundle,
        grids=np.asarray(grids, dtype=np.int64),
        context_frames=np.asarray(context_frames, dtype=np.int64),
    )

    repo_root = Path(__file__).resolve().parent.parent.parent
    payload = (
        "import os, sys; sys.path.insert(0, os.environ['AMBIX_REPO_ROOT']); "
        "from imas_ambix.worldmodel.gate_cohort import _screen_decode_phase; "
        "_screen_decode_phase(os.environ['AMBIX_TOKEN_BUNDLE'], "
        "os.environ['AMBIX_STATS_BUNDLE'], os.environ['AMBIX_DECODE_DEVICE'], "
        "int(os.environ['AMBIX_DECODE_BATCH']))"
    )
    env = dict(os.environ)
    env["AMBIX_REPO_ROOT"] = str(repo_root)
    env["AMBIX_TOKEN_BUNDLE"] = str(token_bundle)
    env["AMBIX_STATS_BUNDLE"] = str(stats_bundle)
    env["AMBIX_DECODE_DEVICE"] = device
    env["AMBIX_DECODE_BATCH"] = str(int(decode_batch_size))
    logger.info(
        "[gate-cohort] decoding %d candidate GT grids in ONE venv subprocess "
        "(model loaded once)",
        int(np.asarray(grids).shape[0]),
    )
    subprocess.run([str(MAGVIT2_PYTHON), "-c", payload], check=True, env=env)
    if not stats_bundle.exists():
        raise RuntimeError(f"screen decode produced no stats bundle at {stats_bundle}")
    s = np.load(str(stats_bundle))
    out = {
        "mean_brightness": np.asarray(s["mean_brightness"], dtype=np.float64),
        "p99_brightness": np.asarray(s["p99_brightness"], dtype=np.float64),
        "motion": np.asarray(s["motion"], dtype=np.float64),
    }
    # driveability components are written by newer decode phases; tolerate an
    # older stats bundle that lacks them (brightness/motion screen still works).
    if "centroid_path" in s:
        out["centroid_path"] = np.asarray(s["centroid_path"], dtype=np.float64)
    if "centroid_profile" in s:
        out["centroid_profile"] = np.asarray(s["centroid_profile"], dtype=np.float64)
    return out


def _apply_gt_stats(rec, mean_bri, p99_bri, motion, thresholds):
    """Fill a record's GT brightness/motion stats from precomputed values + score."""
    rec.mean_brightness = float(mean_bri)
    rec.p99_brightness = float(p99_bri)
    rec.transient_motion = float(motion)
    fails = []
    if rec.mean_brightness < thresholds.min_brightness:
        fails.append(
            f"brightness {rec.mean_brightness:.1f}<{thresholds.min_brightness}"
        )
    if rec.transient_motion < thresholds.min_transient_motion:
        fails.append(
            f"motion {rec.transient_motion:.1f}<{thresholds.min_transient_motion}"
        )
    if rec.plan_variation < thresholds.min_plan_variation:
        fails.append(
            f"plan_var {rec.plan_variation:.2e}<{thresholds.min_plan_variation}"
        )
    if rec.n_streams < thresholds.min_streams:
        fails.append(f"streams {rec.n_streams}<{thresholds.min_streams}")
    rec.passed = not fails
    rec.reason = "pass" if rec.passed else "; ".join(fails)
    return rec


def _select_cohort(records, thresholds, target_size):
    """Re-score every record at *thresholds* and keep up to *target_size* passers.

    Re-scoring uses the cached GT stats on each record, so threshold relaxation
    NEVER re-decodes.  Returns the kept shot-id list (ascending, deterministic).
    """
    cohort: list[int] = []
    for rec in records:
        if not rec.assemblable:
            continue
        _apply_gt_stats(
            rec,
            rec.mean_brightness,
            rec.p99_brightness,
            rec.transient_motion,
            thresholds,
        )
        if rec.passed and len(cohort) < target_size:
            cohort.append(rec.shot_id)
    return cohort


def screen_candidate(shot_id, cfg, *, camera, token_root, thresholds, device, work_dir):
    """Assemble + GT-decode ONE candidate and screen it (single-shot path).

    Convenience wrapper over :func:`_assemble_for_screen` + a one-model-load GT
    decode + :func:`_score_gt`.  The batched builder uses the split helpers
    directly; this single-shot path is kept for tests / ad-hoc use.
    """
    rec, store, ctx = _assemble_for_screen(
        shot_id, cfg, camera=camera, token_root=token_root
    )
    if store is None:
        return rec
    stats = _run_screen_decode(
        store[None, ...], np.asarray([ctx]), work_dir=Path(work_dir), device=device
    )
    return _apply_gt_stats(
        rec,
        stats["mean_brightness"][0],
        stats["p99_brightness"][0],
        stats["motion"][0],
        thresholds,
    )


def build_screened_cohort(
    cfg,
    *,
    camera: str,
    token_root: Path,
    manifest_path: str | Path,
    device: str,
    out_json: str | Path = DEFAULT_COHORT_PATH,
    thresholds: ScreenThresholds | None = None,
    candidate_cap: int = 60,
    target_size: int = 30,
    work_dir: Path | None = None,
) -> dict:
    """Build, leakage-guard, and persist the screened eval-only cohort.

    Enumerates train-disjoint candidates, assembles each window (CPU), STACKS the
    GT camera token grids and decodes them in a SINGLE venv subprocess (VQModel
    loaded once for the whole cohort — AGENTS.md §2b; no per-chunk reload), screens
    on brightness/motion/plan-variation/stream-count, and keeps the passing shots
    up to ``target_size``.  If too few pass at the default brightness/motion gates,
    the thresholds are RELAXED in steps toward their floors (re-scoring the cached
    stats — never re-decoding) until ``target_size`` is reached or the floors are
    hit; the thresholds actually applied are recorded in the JSON.  ASSERTS the
    kept cohort is disjoint from the training manifest, then writes ``out_json`` =
    ``{"cohort": [...ids], "thresholds": {...}, "per_shot": [...screens],
    "summary": {...}}``.  Returns the summary.
    """
    import tempfile  # noqa: PLC0415

    base_thresholds = thresholds or ScreenThresholds()
    train_ids = training_shot_ids(manifest_path)
    candidates = enumerate_candidate_shots(
        token_root=Path(token_root),
        camera=camera,
        train_ids=train_ids,
        cap=candidate_cap,
    )
    logger.info(
        "screening %d train-disjoint candidates (cap %d) against %d training shots",
        len(candidates),
        candidate_cap,
        len(train_ids),
    )
    wd = Path(work_dir or tempfile.mkdtemp(prefix="gate-cohort-screen-"))

    # 1) assemble all candidate windows (CPU); collect the GT store grids to decode.
    screens: list[CohortShotScreen] = []
    decode_recs: list[CohortShotScreen] = []
    decode_grids: list[np.ndarray] = []
    decode_ctx: list[int] = []
    for sid in candidates:
        rec, store, ctx = _assemble_for_screen(
            sid, cfg, camera=camera, token_root=Path(token_root)
        )
        screens.append(rec)
        if store is not None:
            decode_recs.append(rec)
            decode_grids.append(np.asarray(store, dtype=np.int64))
            decode_ctx.append(int(ctx))
    logger.info(
        "assembled %d/%d candidates; decoding GT for all %d in ONE model load",
        len(decode_recs),
        len(candidates),
        len(decode_recs),
    )

    # 2) decode ALL assemblable GT grids in ONE subprocess (model loaded once).
    #    n_frames is fixed by the window config, so every grid is (F,16,16) and
    #    they stack to (N,F,16,16) for a single decode pass.
    if decode_recs:
        stack = np.stack(decode_grids, axis=0)  # (N, F, 16, 16)
        stats = _run_screen_decode(
            stack, np.asarray(decode_ctx), work_dir=wd, device=device
        )
        for j, rec in enumerate(decode_recs):
            # cache the GT stats on the record (default-threshold scoring below).
            _apply_gt_stats(
                rec,
                stats["mean_brightness"][j],
                stats["p99_brightness"][j],
                stats["motion"][j],
                base_thresholds,
            )
            logger.info(
                "screen shot %s: passed=%s (%s) bri=%.1f motion=%.1f streams=%d",
                rec.shot_id,
                rec.passed,
                rec.reason,
                rec.mean_brightness,
                rec.transient_motion,
                rec.n_streams,
            )

    # 3) select the cohort at the default gates; RELAX brightness/motion in steps
    #    toward their floors if too few pass (no re-decode — re-score cached stats).
    applied = base_thresholds
    cohort = _select_cohort(decode_recs, applied, target_size)
    n_relax = 0
    while len(cohort) < target_size and not applied.at_floor():
        applied = applied.relax_step()
        n_relax += 1
        cohort = _select_cohort(decode_recs, applied, target_size)
        logger.info(
            "relaxed thresholds (step %d): bri>=%.1f motion>=%.1f -> %d pass",
            n_relax,
            applied.min_brightness,
            applied.min_transient_motion,
            len(cohort),
        )
    # re-score every record at the FINAL applied thresholds so per_shot.reason
    # reflects what was actually used.
    for rec in decode_recs:
        _apply_gt_stats(
            rec, rec.mean_brightness, rec.p99_brightness, rec.transient_motion, applied
        )

    # binding leakage guard BEFORE persisting.
    assert_disjoint(cohort, train_ids)

    summary = {
        "n_candidates_scanned": len(screens),
        "n_assemblable": sum(1 for r in screens if r.assemblable),
        "n_passed": len(cohort),
        "cohort_size": len(cohort),
        "target_size": target_size,
        "candidate_cap": candidate_cap,
        "camera": camera,
        "n_relaxation_steps": n_relax,
        "thresholds_default": asdict(base_thresholds),
        "thresholds": asdict(applied),
    }
    payload = {
        "cohort": cohort,
        "thresholds": asdict(applied),
        "thresholds_default": asdict(base_thresholds),
        "per_shot": [r.to_dict() for r in screens],
        "summary": summary,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(
        "screened cohort -> %s : %d/%d candidates pass",
        out_json,
        len(cohort),
        len(screens),
    )
    return summary


#: Default location the DRIVEABLE-enriched cohort is written to (a NEW file —
#: distinct from :data:`DEFAULT_COHORT_PATH` so the screened cohort the baked-in
#: eval reads is never overwritten).
DEFAULT_DRIVEABLE_COHORT_PATH = Path(
    "/work/projects/imas_gpu/worldmodel/gate_cohort_driveable.json"
)


def _assemble_for_driveable(shot_id, cfg, *, camera, token_root):
    """Assemble one candidate (CPU) + extract its demanded command-change profile.

    Like :func:`_assemble_for_screen` but additionally returns the per-step
    demanded command-change profile (model-free; from the actuator plan only) used
    for the temporal-association proxy.  Returns
    ``(record, store_frames, ctx, command_profile)``.
    """
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _assemble_heldout,
        _plan_variation,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        recording_time_span_s,
    )

    span_s = recording_time_span_s(int(shot_id), camera=camera, token_root=token_root)
    try:
        sample = _assemble_heldout(
            int(shot_id), cfg, camera=camera, token_root=token_root
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return (
            CohortShotScreen(
                shot_id=int(shot_id),
                assemblable=False,
                passed=False,
                reason=f"unassemblable: {exc!r}",
                recording_span_s=float(span_s or 0.0),
            ),
            None,
            0,
            np.zeros((0,), dtype=np.float64),
        )

    base = sample.signal.base
    ctx = int(base.context_frames)
    ft = np.asarray(base.frame_time, dtype=np.float64)
    stream_names = list(sample.signals.keys())
    cmd_profile = _command_change_profile(
        sample.actuator.values, sample.actuator.missing
    )
    rec = CohortShotScreen(
        shot_id=int(shot_id),
        assemblable=True,
        passed=False,
        recording_span_s=float(span_s or 0.0),
        achieved_window_span_s=float(ft[-1] - ft[0]) if ft.size >= 2 else 0.0,
        context_frames=ctx,
        n_frames=int(base.n_frames),
        plan_variation=float(_plan_variation(sample)),
        command_change_total=float(cmd_profile.sum()),
        n_streams=len(stream_names),
        stream_names=stream_names,
    )
    return rec, base.store_frames(), ctx, cmd_profile


def build_driveable_cohort(
    cfg,
    *,
    camera: str,
    token_root: Path,
    manifest_path: str | Path,
    device: str,
    out_json: str | Path = DEFAULT_DRIVEABLE_COHORT_PATH,
    thresholds: DriveabilityThresholds | None = None,
    candidate_cap: int = 400,
    target_size: int = 18,
    work_dir: Path | None = None,
) -> dict:
    """Build a DRIVEABLE-ENRICHED, train-disjoint eval-only cohort (model-free).

    The screened cohort (:func:`build_screened_cohort`) mixes genuinely-driveable
    shots with shots whose plasma cannot respond to commands in principle, so its
    cohort-mean UNDERSTATES real controllability.  This sibling builder enriches
    for the driveable regime: it enumerates the same train-disjoint candidates,
    assembles each window (CPU), STACKS + decodes the GT camera grids in a SINGLE
    venv subprocess (VQ loaded once — AGENTS.md §2b), then RANKS the candidates by
    a physical :func:`driveability_score` and keeps the top ``target_size`` shots
    that clear the (permissive) :class:`DriveabilityThresholds` eligibility floors.

    Anti-circularity (binding): every quantity that drives selection is a property
    of the GROUND-TRUTH frames + the DEMANDED commands ONLY — never of a
    world-model's response — so the cohort cannot be biased toward shots the model
    already drives.

    Writes ``out_json`` (a NEW file — :data:`DEFAULT_DRIVEABLE_COHORT_PATH` —
    distinct from the screened cohort the baked-in eval reads) with the per-shot
    driveability components + thresholds, after the binding leakage guard.  Returns
    the summary.
    """
    import tempfile  # noqa: PLC0415

    thr = thresholds or DriveabilityThresholds()
    train_ids = training_shot_ids(manifest_path)
    candidates = enumerate_candidate_shots(
        token_root=Path(token_root),
        camera=camera,
        train_ids=train_ids,
        cap=candidate_cap,
    )
    logger.info(
        "driveable-screening %d train-disjoint candidates (cap %d) vs %d train shots",
        len(candidates),
        candidate_cap,
        len(train_ids),
    )
    wd = Path(work_dir or tempfile.mkdtemp(prefix="gate-cohort-driveable-"))

    # 1) assemble all candidate windows (CPU); collect GT grids + command profiles.
    screens: list[CohortShotScreen] = []
    decode_recs: list[CohortShotScreen] = []
    decode_grids: list[np.ndarray] = []
    decode_ctx: list[int] = []
    cmd_profiles: list[np.ndarray] = []
    for sid in candidates:
        rec, store, ctx, cmd_prof = _assemble_for_driveable(
            sid, cfg, camera=camera, token_root=Path(token_root)
        )
        screens.append(rec)
        if store is not None:
            decode_recs.append(rec)
            decode_grids.append(np.asarray(store, dtype=np.int64))
            decode_ctx.append(int(ctx))
            cmd_profiles.append(cmd_prof)
    logger.info(
        "assembled %d/%d candidates; decoding GT for all %d in ONE model load",
        len(decode_recs),
        len(candidates),
        len(decode_recs),
    )

    # 2) decode ALL assemblable GT grids in ONE subprocess (model loaded once) —
    #    yields per-shot brightness/motion + GT centroid path + displacement profile.
    if decode_recs:
        stack = np.stack(decode_grids, axis=0)  # (N, F, 16, 16)
        stats = _run_screen_decode(
            stack, np.asarray(decode_ctx), work_dir=wd, device=device
        )
        screen_thr = ScreenThresholds(
            min_brightness=thr.min_brightness,
            min_transient_motion=thr.min_transient_motion,
            min_plan_variation=thr.min_plan_variation,
            min_streams=thr.min_streams,
        )
        cen_path = stats.get("centroid_path")
        cen_prof = stats.get("centroid_profile")
        for j, rec in enumerate(decode_recs):
            _apply_gt_stats(
                rec,
                stats["mean_brightness"][j],
                stats["p99_brightness"][j],
                stats["motion"][j],
                screen_thr,
            )
            if cen_path is not None:
                rec.centroid_path_px = float(cen_path[j])
            prof_j = cen_prof[j] if cen_prof is not None else None
            _apply_driveability(
                rec, command_profile=cmd_profiles[j], centroid_profile=prof_j
            )
            logger.info(
                "driveable shot %s: drive=%.4f plan_var=%.2f motion=%.1f "
                "cen_path=%.1f assoc=%.2f bri=%.1f streams=%d",
                rec.shot_id,
                rec.driveability_score,
                rec.plan_variation,
                rec.transient_motion,
                rec.centroid_path_px,
                rec.temporal_association,
                rec.mean_brightness,
                rec.n_streams,
            )

    # 3) keep the TOP-target_size eligible candidates by driveability score.
    for rec in decode_recs:
        fails = thr.eligibility_fails(rec)
        rec.passed = not fails
        rec.reason = "eligible" if rec.passed else "; ".join(fails)
    cohort = _select_driveable_cohort(decode_recs, thr, target_size)

    # binding leakage guard BEFORE persisting.
    assert_disjoint(cohort, train_ids)

    cohort_set = set(cohort)
    for rec in decode_recs:
        if rec.shot_id in cohort_set:
            rec.passed = True
            rec.reason = "kept (top driveability)"
        elif not rec.reason or rec.reason == "eligible":
            rec.reason = "eligible (not in top-N)"

    kept = [r for r in screens if r.shot_id in cohort_set]
    summary = {
        "mode": "driveable",
        "n_candidates_scanned": len(screens),
        "n_assemblable": sum(1 for r in screens if r.assemblable),
        "n_eligible": sum(1 for r in decode_recs if not thr.eligibility_fails(r)),
        "n_passed": len(cohort),
        "cohort_size": len(cohort),
        "target_size": target_size,
        "candidate_cap": candidate_cap,
        "camera": camera,
        "thresholds": asdict(thr),
        "kept_driveability_min": (
            min(r.driveability_score for r in kept) if kept else 0.0
        ),
        "kept_driveability_max": (
            max(r.driveability_score for r in kept) if kept else 0.0
        ),
    }
    payload = {
        "cohort": cohort,
        "mode": "driveable",
        "thresholds": asdict(thr),
        "per_shot": [r.to_dict() for r in screens],
        "summary": summary,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(
        "driveable cohort -> %s : top %d of %d eligible (by driveability score)",
        out_json,
        len(cohort),
        summary["n_eligible"],
    )
    return summary


def load_cohort(path: str | Path) -> list[int]:
    """Read the cohort shot ids from a ``build_screened_cohort`` JSON."""
    data = json.loads(Path(path).read_text())
    return [int(s) for s in data.get("cohort", [])]


__all__ = [
    "DEFAULT_COHORT_PATH",
    "DEFAULT_DRIVEABLE_COHORT_PATH",
    "DEFAULT_TOKEN_ROOT",
    "CohortShotScreen",
    "DriveabilityThresholds",
    "ScreenThresholds",
    "assert_disjoint",
    "build_driveable_cohort",
    "build_screened_cohort",
    "driveability_score",
    "enumerate_candidate_shots",
    "load_cohort",
    "screen_candidate",
    "training_shot_ids",
]


# Re-export the CPU-testable screen-scoring helpers (not GPU-dependent) so the
# unit tests can exercise the gate logic without a VQ decode.
__all__ += [
    "_apply_gt_stats",
    "_apply_driveability",
    "_centroid_path_length",
    "_centroid_displacement_profile",
    "_command_change_profile",
    "_gt_window_stats",
    "_score_gt",
    "_select_cohort",
    "_select_driveable_cohort",
    "_temporal_association",
]
