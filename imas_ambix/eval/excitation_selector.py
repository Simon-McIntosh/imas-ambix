"""Model-independent excitation-window selector + the excited held-out cohort.

Why this exists (the identifiability device)
--------------------------------------------
A controllability gate asks: does the demanded command MOVE the predicted state?
On a flat-top window the actuators barely move, so the command->state map is
**under-identified** — the gate cannot tell a driveable model from a model that
ignores its plan, and it FALSELY fails (the causal-under-identification confound,
control-conditioning survey §4).  The fix is to evaluate ONLY on windows where
the command *demonstrably should* move the state: windows with genuine actuator
excitation — coil-current ramps ``|dI/dt|``, ramp-up / ramp-down phases, and
step changes in the demanded plan.

This module scores a (shot, window) by GENUINE actuator excitation, selects the
top-excited windows, and — critically — assembles the **excited held-out
cohort**: held-out shots whose best window clears an excitation threshold.  That
cohort is the device that breaks the confound, and it is what the powered ΔN-M
controllability gate runs on.  A binding LEAKAGE AUDIT asserts the excited cohort
is disjoint from the training shots and that the locked held-out family
18502-18505 is preserved as held-out.

Model independence (binding)
----------------------------
Every quantity scored here is a property of the DEMANDED actuator plan and the
COIL/PF currents ONLY — never of any world-model's response, and never of the
camera pixels.  Excitation is "did the operator drive the machine?", measured on
the plan/coil streams.  No model rollout is ever consulted, so the cohort cannot
be biased toward windows a particular model already drives.

Pure-numpy core vs live wrappers
--------------------------------
The scoring logic is factored into pure-numpy functions
(:func:`coil_ramp_profile`, :func:`window_excitation_score`,
:func:`plan_change_excitation`) that take raw arrays and are fully CPU-testable
on a synthetic plan/coil stream with no ``/work`` access.  Thin wrappers
(:func:`score_shot_windows`, :func:`select_excited_windows`,
:func:`build_excited_heldout_cohort`) drive the live on-disk primitives in
:mod:`imas_ambix.worldmodel.excitation_corpus` /
:mod:`imas_ambix.worldmodel.actuator_plan` when ``/work`` is reachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.worldmodel.actuator_plan import (
    coil_current_channel_indices,
    plasma_current_channel_index,
)
from imas_ambix.worldmodel.excitation_corpus import (
    DEFAULT_HELD_OUT,
    CuratedWindow,
    enumerate_shot_windows,
)
from imas_ambix.worldmodel.gate_cohort import assert_disjoint, training_shot_ids

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

#: The locked held-out family — whole pulses kept OUT of training and used as the
#: controllability probe set.  Re-exported from the curation module so the audit
#: has a single source of truth (and the "18502-18505 preserved as held-out"
#: invariant cannot drift between the two modules).
LOCKED_HELD_OUT: tuple[int, ...] = tuple(int(s) for s in DEFAULT_HELD_OUT)

#: Default excitation threshold (summed coil ``|dI/dt|``, physical units) that a
#: window must clear to count as EXCITED rather than flat-top.  Matches the
#: curation corpus's ``min_excitation`` floor (1e3) so "excited" means the same
#: thing for the eval cohort as it does for the training corpus.
DEFAULT_EXCITATION_THRESHOLD: float = 1.0e3


# ---------------------------------------------------------------------------
# Pure-numpy excitation core (CPU-testable; no /work, no model, no pixels)
# ---------------------------------------------------------------------------


def coil_ramp_profile(coil_currents: np.ndarray, frame_time: np.ndarray) -> np.ndarray:
    """Per-frame summed coil ramp rate ``|dI/dt|`` over the (possibly uneven) grid.

    ``coil_currents`` is ``(n, k)`` physical coil currents (one column per present
    coil channel); ``frame_time`` is the ``(n,)`` timestamp axis (need not be
    uniform).  Returns the ``(n,)`` per-frame summed ``|dI/dt|`` (the last frame
    repeats the previous rate, having no successor).  This is the level-INVARIANT
    excitation signal — a flat-but-large current scores ~0, a ramping current
    scores high — factored out of :func:`find_excitation_window` so it can be
    exercised on a synthetic stream.
    """
    coil = np.asarray(coil_currents, dtype=np.float64)
    ftime = np.asarray(frame_time, dtype=np.float64)
    if coil.ndim == 1:
        coil = coil[:, None]
    n = coil.shape[0]
    if n < 2:
        return np.zeros((n,), dtype=np.float64)
    dt = np.diff(ftime)
    dt = np.where(dt > 0, dt, np.nan)
    didt = np.full_like(coil, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.diff(coil, axis=0) / dt[:, None]
    didt[:-1] = rate
    didt[-1] = rate[-1] if rate.shape[0] else 0.0
    return np.abs(np.where(np.isfinite(didt), didt, 0.0)).sum(axis=1)


def window_excitation_score(ramp_profile: np.ndarray) -> float:
    """Time-mean of a window's per-frame summed coil ``|dI/dt|`` (the score).

    The single scalar that ranks windows by persistent coil excitation: the mean
    over the window of :func:`coil_ramp_profile`.  Higher == more persistently
    excited.  ``0.0`` for an empty window.
    """
    r = np.asarray(ramp_profile, dtype=np.float64)
    r = r[np.isfinite(r)]
    return float(r.mean()) if r.size else 0.0


def plan_change_excitation(plan_values: np.ndarray, plan_missing: np.ndarray) -> float:
    """Total demanded-command change over a window (model-free plan excitation).

    ``plan_values`` is the ``(n_plan, C)`` (normalised) demanded actuator surface
    and ``plan_missing`` its ``(n_plan, C)`` missing flag.  Returns the summed L1
    change of the demanded vector between consecutive plan steps over the present
    channels — a flat plan scores ~0, a stepped/ramped plan scores high.  This is
    the complementary, plan-side excitation measure to the coil ``|dI/dt|`` (it
    captures gas-puff / NBI toggles that are not coil currents).  Property of the
    DEMANDED commands only.
    """
    vals = np.asarray(plan_values, dtype=np.float64)
    miss = np.asarray(plan_missing, dtype=np.float64)
    if vals.ndim != 2 or vals.shape[0] < 2:
        return 0.0
    present = miss.mean(axis=0) < 1.0 if miss.size else np.ones(vals.shape[1], bool)
    if not bool(np.asarray(present).any()):
        return 0.0
    return float(np.abs(np.diff(vals[:, present], axis=0)).sum())


def is_excited(score: float, threshold: float = DEFAULT_EXCITATION_THRESHOLD) -> bool:
    """True when an excitation score clears the excited/flat-top threshold."""
    return bool(np.isfinite(score) and float(score) >= float(threshold))


# ---------------------------------------------------------------------------
# Window scoring over a shot/window stream (synthetic-friendly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredWindow:
    """One scored time-window: which (shot, start) and its excitation score."""

    shot_id: int
    start_frame: int
    excitation_score: float
    max_abs_ip: float = 0.0
    present_fraction: float = 0.0
    phase: str = ""
    n_frames: int = 0
    frame_stride: int = 1

    @property
    def excited(self) -> bool:
        return is_excited(self.excitation_score)


def score_windows_from_streams(
    coil_currents: np.ndarray,
    frame_time: np.ndarray,
    *,
    span: int,
    shot_id: int = 0,
    stride: int | None = None,
) -> list[ScoredWindow]:
    """Score every sliding ``span``-frame window of one coil/frame-time stream.

    Pure-numpy, CPU-only, no ``/work``: slides a ``span``-frame window across the
    ``(n, k)`` coil-current series at ``stride`` frames (default ``span // 2``),
    scoring each on the time-mean summed coil ``|dI/dt|``
    (:func:`window_excitation_score` of :func:`coil_ramp_profile`).  Returns the
    scored windows in ascending start-frame order.  This is the model-independent
    selector core — the live wrapper :func:`score_shot_windows` feeds it the
    on-disk coil stream.
    """
    coil = np.asarray(coil_currents, dtype=np.float64)
    if coil.ndim == 1:
        coil = coil[:, None]
    ftime = np.asarray(frame_time, dtype=np.float64)
    n = coil.shape[0]
    span = int(span)
    if span < 2 or n < span:
        return []
    ramp = coil_ramp_profile(coil, ftime)
    step = int(stride) if stride else max(1, span // 2)
    out: list[ScoredWindow] = []
    for start in range(0, n - span + 1, step):
        sl = slice(start, start + span)
        out.append(
            ScoredWindow(
                shot_id=int(shot_id),
                start_frame=int(start),
                excitation_score=window_excitation_score(ramp[sl]),
                n_frames=span,
                frame_stride=1,
            )
        )
    return out


def select_excited_windows(
    scored: Sequence[ScoredWindow],
    *,
    threshold: float = DEFAULT_EXCITATION_THRESHOLD,
    limit: int | None = None,
) -> list[ScoredWindow]:
    """Keep the windows that clear ``threshold``, most-excited first.

    Filters a scored-window list to the EXCITED windows (score ``>= threshold``)
    and returns them sorted by descending excitation (then ascending shot id,
    start frame for stability), optionally capped at ``limit``.  This is the
    "curate the training distribution toward excitation" / "pick the excited
    probe windows" operation, model-independent.
    """
    excited = [w for w in scored if is_excited(w.excitation_score, threshold)]
    excited.sort(key=lambda w: (-w.excitation_score, w.shot_id, w.start_frame))
    if limit is not None:
        excited = excited[: int(limit)]
    return excited


# ---------------------------------------------------------------------------
# Live wrappers (drive the on-disk primitives when /work is reachable)
# ---------------------------------------------------------------------------


def score_shot_windows(
    shot_id: int,
    *,
    token_root: Path | None = None,
    min_excitation: float = DEFAULT_EXCITATION_THRESHOLD,
    **kwargs,
) -> list[ScoredWindow]:
    """Score a real shot's windows via the on-disk coil-excitation primitive.

    Tiles the pulse with :func:`excitation_corpus.enumerate_shot_windows` (the
    working coil ``|dI/dt|`` window scorer, which already loads the actuator
    vector ONCE and scores plasma-present windows), and maps its
    :class:`CuratedWindow` rows to :class:`ScoredWindow`.  Requires ``/work`` (the
    token store + level-1 actuator records); returns ``[]`` when unreadable.  All
    excitation comes from the coil/plan streams — model-independent.
    """
    rows: list[CuratedWindow] = enumerate_shot_windows(
        int(shot_id),
        token_root=token_root,
        min_excitation=float(min_excitation),
        **kwargs,
    )
    return [
        ScoredWindow(
            shot_id=int(r.shot_id),
            start_frame=int(r.start_frame),
            excitation_score=float(r.excitation_score),
            max_abs_ip=float(r.max_abs_ip),
            present_fraction=float(r.present_fraction),
            phase=str(r.phase),
            n_frames=int(r.n_frames),
            frame_stride=int(r.frame_stride),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# The excited held-out cohort + the leakage audit (the gate deliverable)
# ---------------------------------------------------------------------------


@dataclass
class ExcitedCohort:
    """The excited held-out cohort + its excitation justification and audit.

    Attributes
    ----------
    shot_ids:
        The excited held-out shots — held-out (train-disjoint) shots whose best
        window clears the excitation threshold.  The powered ΔN-M gate runs on
        these.
    threshold:
        The excitation threshold (summed coil ``|dI/dt|``) separating excited
        from flat-top.
    per_shot_best_score:
        ``{shot_id: best_window_excitation_score}`` over every candidate shot
        scored (excited + rejected) — the score distribution that JUSTIFIES the
        threshold.
    rejected_flat:
        Candidate held-out shots whose best window did NOT clear the threshold
        (the flat-top reject) — reported so the excited/flat-top separation is
        visible.
    locked_held_out:
        The locked held-out family (18502-18505) — asserted to remain held-out.
    disjoint:
        True when the leakage audit passed (cohort ∩ training == ∅).
    n_train_shots:
        Number of distinct training shots the audit checked against.
    """

    shot_ids: list[int] = field(default_factory=list)
    threshold: float = DEFAULT_EXCITATION_THRESHOLD
    per_shot_best_score: dict = field(default_factory=dict)
    rejected_flat: list[int] = field(default_factory=list)
    locked_held_out: tuple[int, ...] = LOCKED_HELD_OUT
    disjoint: bool = False
    n_train_shots: int = 0

    def score_distribution_summary(self) -> dict:
        """Summary stats of the per-shot best-score distribution (for the report)."""
        scores = np.asarray(
            [float(v) for v in self.per_shot_best_score.values()], dtype=np.float64
        )
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return {"n": 0}
        return {
            "n": int(scores.size),
            "min": float(scores.min()),
            "p50": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
            "max": float(scores.max()),
            "n_excited": int((scores >= float(self.threshold)).sum()),
            "n_flat": int((scores < float(self.threshold)).sum()),
            "threshold": float(self.threshold),
        }


def assemble_excited_cohort(
    candidate_scores: dict,
    train_ids,
    *,
    threshold: float = DEFAULT_EXCITATION_THRESHOLD,
    locked_held_out: Sequence[int] = LOCKED_HELD_OUT,
    target_size: int | None = None,
) -> ExcitedCohort:
    """Assemble + leakage-audit the excited held-out cohort from per-shot scores.

    Pure logic (CPU-testable): given ``candidate_scores`` =
    ``{shot_id: best_window_excitation_score}`` over the HELD-OUT candidate shots
    and the set of training shot ids, keeps the excited shots (best score ``>=
    threshold``), runs the binding leakage audit
    (:func:`gate_cohort.assert_disjoint` — RAISES if any excited shot leaks into
    the training set), and asserts every locked held-out shot remains held-out
    (i.e. not in the training set).  Excited shots are returned most-excited
    first, optionally capped at ``target_size``.

    The split between "score the windows" (live, needs ``/work``) and "assemble +
    audit the cohort" (this function, pure) is deliberate: the audit + threshold
    logic is fully exercised on synthetic scores, and the same function audits the
    live build.
    """
    train_set = {int(s) for s in train_ids}
    locked = tuple(int(s) for s in locked_held_out)

    # the locked held-out family must NOT have leaked into training.
    locked_leak = sorted(set(locked) & train_set)
    assert not locked_leak, (
        f"locked held-out family leaked into training: {locked_leak} — "
        "18502-18505 must be preserved as held-out"
    )

    best = {int(k): float(v) for k, v in candidate_scores.items()}
    excited = [s for s, sc in best.items() if is_excited(sc, threshold)]
    rejected = sorted(s for s, sc in best.items() if not is_excited(sc, threshold))
    # most-excited first; deterministic shot-id tie-break.
    excited.sort(key=lambda s: (-best[s], s))
    if target_size is not None:
        excited = excited[: int(target_size)]

    # binding leakage audit — RAISES if any excited cohort shot is a training shot.
    assert_disjoint(excited, train_set, manifest_label="training shot set")

    return ExcitedCohort(
        shot_ids=excited,
        threshold=float(threshold),
        per_shot_best_score=best,
        rejected_flat=rejected,
        locked_held_out=locked,
        disjoint=True,
        n_train_shots=len(train_set),
    )


def build_excited_heldout_cohort(
    candidate_shots: Sequence[int],
    *,
    manifest_path: str | Path,
    token_root: Path | None = None,
    threshold: float = DEFAULT_EXCITATION_THRESHOLD,
    locked_held_out: Sequence[int] = LOCKED_HELD_OUT,
    min_excitation: float = DEFAULT_EXCITATION_THRESHOLD,
    target_size: int | None = None,
    **window_kwargs,
) -> ExcitedCohort:
    """Build the excited held-out cohort from LIVE on-disk shot streams.

    For each candidate shot (which MUST be train-disjoint held-out shots), scores
    its windows via :func:`score_shot_windows` (the on-disk coil-excitation
    primitive) and records its best window score, then hands the per-shot best
    scores to :func:`assemble_excited_cohort` for the threshold cut + the binding
    leakage audit against the training manifest's shot set
    (:func:`gate_cohort.training_shot_ids`).

    Requires ``/work``.  The CPU-testable core is :func:`assemble_excited_cohort`;
    this wrapper only adds the live window scoring.
    """
    train_ids = training_shot_ids(manifest_path)
    best_scores: dict = {}
    for sid in candidate_shots:
        windows = score_shot_windows(
            int(sid),
            token_root=token_root,
            min_excitation=float(min_excitation),
            **window_kwargs,
        )
        best_scores[int(sid)] = max((w.excitation_score for w in windows), default=0.0)
    return assemble_excited_cohort(
        best_scores,
        train_ids,
        threshold=threshold,
        locked_held_out=locked_held_out,
        target_size=target_size,
    )


def coil_channel_columns() -> list[int]:
    """The coil-current actuator columns (re-export — the excitation channels)."""
    return coil_current_channel_indices()


def ip_channel_column() -> int | None:
    """The plasma-current column (the response/presence signal, NOT a command)."""
    return plasma_current_channel_index()


__all__ = [
    "DEFAULT_EXCITATION_THRESHOLD",
    "LOCKED_HELD_OUT",
    "ExcitedCohort",
    "ScoredWindow",
    "assemble_excited_cohort",
    "build_excited_heldout_cohort",
    "coil_channel_columns",
    "coil_ramp_profile",
    "ip_channel_column",
    "is_excited",
    "plan_change_excitation",
    "score_shot_windows",
    "score_windows_from_streams",
    "select_excited_windows",
    "window_excitation_score",
]
