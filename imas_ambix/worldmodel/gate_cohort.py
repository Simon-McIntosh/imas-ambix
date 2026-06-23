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

GPU-safety (AGENTS.md §2b): the GT decode loads the frozen VQ ONCE and decodes
all candidates in batched VQ passes inside the decode subprocess; no per-shot
model reload.  The candidate enumeration + train-disjoint set + leakage guard
are pure-numpy / filesystem and CPU-testable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

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


def screen_candidate(shot_id, cfg, *, camera, token_root, thresholds, device, work_dir):
    """Assemble + GT-decode one candidate and screen it against ``thresholds``.

    Returns a :class:`CohortShotScreen`.  Unassemblable shots (recording shorter
    than the horizon, unreadable cadence) are returned with ``assemblable=False``
    so the caller records WHY each candidate was dropped.  The GT decode reuses
    the same one-VQ-pass :func:`decode_roles` path the eval uses.
    """
    from imas_ambix.worldmodel.control_falsification import (
        decode_roles,  # noqa: PLC0415
    )
    from imas_ambix.worldmodel.control_guidance import _to_gray_f64  # noqa: PLC0415
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
        return CohortShotScreen(
            shot_id=int(shot_id),
            assemblable=False,
            passed=False,
            reason=f"unassemblable: {exc!r}",
            recording_span_s=float(span_s or 0.0),
        )

    base = sample.signal.base
    ctx = int(base.context_frames)
    ft = np.asarray(base.frame_time, dtype=np.float64)
    stream_names = list(sample.signals.keys())
    plan_var = _plan_variation(sample)

    store = base.store_frames()  # (F, 16, 16) store ids
    decoded = decode_roles(
        {"gt": store},
        [{"role": "gt"}],
        work_dir=Path(work_dir) / f"shot{shot_id}",
        device=device,
    )
    gt_gray = _to_gray_f64(decoded["gt"])  # (F, H, W)
    fwin = gt_gray[ctx:]
    mean_bri = float(fwin.mean()) if fwin.size else 0.0
    p99 = float(np.percentile(fwin, 99)) if fwin.size else 0.0
    motion = float(np.abs(np.diff(fwin, axis=0)).mean()) if fwin.shape[0] >= 2 else 0.0

    rec = CohortShotScreen(
        shot_id=int(shot_id),
        assemblable=True,
        passed=False,
        recording_span_s=float(span_s or 0.0),
        achieved_window_span_s=float(ft[-1] - ft[0]) if ft.size >= 2 else 0.0,
        context_frames=ctx,
        n_frames=int(base.n_frames),
        mean_brightness=mean_bri,
        p99_brightness=p99,
        transient_motion=motion,
        plan_variation=float(plan_var),
        n_streams=len(stream_names),
        stream_names=stream_names,
    )
    fails = []
    if mean_bri < thresholds.min_brightness:
        fails.append(f"brightness {mean_bri:.1f}<{thresholds.min_brightness}")
    if motion < thresholds.min_transient_motion:
        fails.append(f"motion {motion:.1f}<{thresholds.min_transient_motion}")
    if plan_var < thresholds.min_plan_variation:
        fails.append(f"plan_var {plan_var:.2e}<{thresholds.min_plan_variation}")
    if len(stream_names) < thresholds.min_streams:
        fails.append(f"streams {len(stream_names)}<{thresholds.min_streams}")
    rec.passed = not fails
    rec.reason = "pass" if rec.passed else "; ".join(fails)
    return rec


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

    Enumerates train-disjoint candidates, screens each (assemble + GT-decode +
    brightness/motion/plan-variation/stream-count), keeps the passing shots up to
    ``target_size``, ASSERTS the kept cohort is disjoint from the training
    manifest, and writes ``out_json`` = ``{"cohort": [...ids], "thresholds":
    {...}, "per_shot": [...screens], "summary": {...}}``.  Returns the summary.
    """
    import tempfile  # noqa: PLC0415

    thresholds = thresholds or ScreenThresholds()
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
    screens: list[CohortShotScreen] = []
    cohort: list[int] = []
    for sid in candidates:
        rec = screen_candidate(
            sid,
            cfg,
            camera=camera,
            token_root=Path(token_root),
            thresholds=thresholds,
            device=device,
            work_dir=wd,
        )
        screens.append(rec)
        logger.info(
            "screen shot %s: assemblable=%s passed=%s (%s) "
            "bri=%.1f motion=%.1f streams=%d",
            rec.shot_id,
            rec.assemblable,
            rec.passed,
            rec.reason,
            rec.mean_brightness,
            rec.transient_motion,
            rec.n_streams,
        )
        if rec.passed:
            cohort.append(rec.shot_id)
        if len(cohort) >= target_size:
            break

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
        "thresholds": asdict(thresholds),
    }
    payload = {
        "cohort": cohort,
        "thresholds": asdict(thresholds),
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


def load_cohort(path: str | Path) -> list[int]:
    """Read the cohort shot ids from a ``build_screened_cohort`` JSON."""
    data = json.loads(Path(path).read_text())
    return [int(s) for s in data.get("cohort", [])]


__all__ = [
    "DEFAULT_COHORT_PATH",
    "DEFAULT_TOKEN_ROOT",
    "CohortShotScreen",
    "ScreenThresholds",
    "assert_disjoint",
    "build_screened_cohort",
    "enumerate_candidate_shots",
    "load_cohort",
    "screen_candidate",
    "training_shot_ids",
]
