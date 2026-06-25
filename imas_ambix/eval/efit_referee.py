"""Firewalled referee over EFIT's reconstructed equilibrium geometry.

What this is
------------
The referee reads the EFIT-reconstructed equilibrium **outputs** (magnetic
axis, primary X-point, last-closed-flux-surface boundary, and — where the
store carries it — the q profile) for a shot/time and scores how well a
**model-produced** geometry agrees with them.  Agreement-with-EFIT is the
*weak* tier of the validation honesty hierarchy: a sanity check, never the
bar the world model is trained to.

The firewall (binding, inherited definition)
---------------------------------------------
``firewall = CODE-OUTPUTS-ONLY``.  EFIT plasma-geometry **reconstruction
outputs** (boundary / X-point / ψ / q) are *evaluator-only* — they must never
become a training label or a model input.  SENSOR geometry (where diagnostics
sit / point) is apparatus metadata and is freely usable; it is NOT firewalled
and this module does not gate it.

So this module gates only the reconstruction-output read.  The gate is a
**runtime** guard plus a **static** one:

* Runtime — every read of EFIT outputs goes through :func:`read_efit_geometry`,
  which raises :class:`FirewallViolation` unless it is called inside an explicit
  :func:`evaluator_context`.  An accidental use on a training / model-input path
  therefore fails loudly instead of silently leaking the reconstruction.
* Static — no training / model module imports this module (proven by the test
  suite, which greps the world-model source tree and asserts zero hits).

Reuse, don't reinvent
---------------------
The on-disk equilibrium layout already has a reader,
:mod:`imas_ambix.worldmodel.equilibrium_labels` (itself evaluator-only).  This
referee reuses it read-only: :func:`read_efit_geometry` defaults to
``load_equilibrium_geometry`` (real L2 Zarr store, ``level2_root``-injectable)
but accepts any ``loader`` callable, so tests drive it on synthetic arrays
without ``/work`` access.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from imas_ambix.worldmodel.equilibrium_labels import EquilibriumGeometry


# ---------------------------------------------------------------------------
# Firewall guard
# ---------------------------------------------------------------------------


class FirewallViolation(RuntimeError):  # noqa: N818 - named for the firewall contract, not the Error suffix convention
    """Raised when EFIT reconstruction outputs are read off an evaluator path.

    The EFIT equilibrium reconstruction is evaluator-only.  Reading it outside
    an explicit :func:`evaluator_context` is presumed to be an accidental use
    on a training / model-input code path and is refused loudly so the leak is
    caught at the call site rather than silently contaminating a label.
    """


# Thread-local so a referee read on one evaluation thread cannot accidentally
# open the gate for an unrelated thread (e.g. a concurrent training worker).
_GATE = threading.local()


def _gate_open() -> bool:
    return bool(getattr(_GATE, "depth", 0) > 0)


@contextmanager
def evaluator_context() -> Iterator[None]:
    """Open the firewall gate for the duration of an evaluation block.

    Only inside this context may :func:`read_efit_geometry` read the EFIT
    reconstruction outputs.  Re-entrant (nesting is counted), thread-local, and
    always closed on exit even if the body raises::

        with evaluator_context():
            ref = read_efit_geometry(shot_id, frame_times)
        # read_efit_geometry(...) here would raise FirewallViolation
    """
    _GATE.depth = getattr(_GATE, "depth", 0) + 1
    try:
        yield
    finally:
        _GATE.depth -= 1


# ---------------------------------------------------------------------------
# Gated read of EFIT reconstruction outputs
# ---------------------------------------------------------------------------


def _default_loader(
    shot_id: int,
    frame_times: np.ndarray,
    **kwargs,
) -> EquilibriumGeometry:
    """Default reader: the evaluator-only L2 equilibrium label loader.

    Imported lazily so this module loads without the world-model package (and
    so the static import-guard test sees no top-level coupling).
    """
    from imas_ambix.worldmodel.equilibrium_labels import (  # noqa: PLC0415
        load_equilibrium_geometry,
    )

    return load_equilibrium_geometry(shot_id, frame_times, **kwargs)


def read_efit_geometry(
    shot_id: int,
    frame_times: np.ndarray,
    *,
    loader: Callable[..., EquilibriumGeometry] | None = None,
    **loader_kwargs,
) -> EquilibriumGeometry:
    """Read EFIT-reconstructed equilibrium geometry — **gated**.

    This is the *only* entry point in this module that touches the EFIT
    reconstruction outputs, and it refuses unless called inside
    :func:`evaluator_context`.

    Parameters
    ----------
    shot_id:
        Shot whose EFIT equilibrium reconstruction is read.
    frame_times:
        ``(F,)`` times (seconds) to interpolate the geometry onto.
    loader:
        Optional reader ``(shot_id, frame_times, **kwargs) -> EquilibriumGeometry``.
        Defaults to the real L2-store loader; tests inject a synthetic one so
        no ``/work`` access is needed.
    **loader_kwargs:
        Forwarded to ``loader`` (e.g. ``level2_root`` for the default).

    Returns
    -------
    EquilibriumGeometry
        The 12-D per-frame reference geometry (axis R/Z, primary X-point R/Z,
        8 LCFS control-point radii) in metres, with a per-component finite mask.

    Raises
    ------
    FirewallViolation
        If called outside an :func:`evaluator_context`.
    """
    if not _gate_open():
        raise FirewallViolation(
            "EFIT equilibrium reconstruction outputs are evaluator-only: "
            "read_efit_geometry() must be called inside evaluator_context(). "
            "If this fires on a training or model-input path, that path is "
            "leaking the reconstruction — remove the call, do not widen the gate."
        )
    fn = loader or _default_loader
    return fn(shot_id, frame_times, **loader_kwargs)


# ---------------------------------------------------------------------------
# Judge: score model geometry against the EFIT reference
# ---------------------------------------------------------------------------


@dataclass
class GeometryVerdict:
    """Structured agreement of a candidate geometry vs. the EFIT reference.

    All errors are in metres; ``None`` where the reference component is masked
    (the EFIT reconstruction was undefined, e.g. plasma-off) or the candidate
    omits it, so the metric is genuinely unavailable rather than zero.

    Attributes
    ----------
    axis_error:
        Euclidean distance (m) between candidate and reference magnetic axis.
    xpoint_error:
        Euclidean distance (m) between candidate and reference primary X-point.
    boundary_rms:
        RMS difference (m) of the LCFS control-point radii, over the angles
        present and finite in both candidate and reference.
    n_boundary_points:
        Count of LCFS control points that entered ``boundary_rms``.
    components:
        Per-component absolute error (m) keyed by name, for every component
        finite in both candidate and reference.
    passed:
        Whether every available error is within its tolerance (see
        :func:`judge_geometry`); ``None`` if no component was comparable.
    notes:
        Free-form remarks (e.g. which components were masked / missing).
    """

    axis_error: float | None = None
    xpoint_error: float | None = None
    boundary_rms: float | None = None
    n_boundary_points: int = 0
    components: dict[str, float] = field(default_factory=dict)
    passed: bool | None = None
    notes: list[str] = field(default_factory=list)


# Component name layout shared with equilibrium_labels (kept local so this
# module does not import the world-model package at module load — see firewall).
_AXIS_NAMES = ("axis_R", "axis_Z")
_XPOINT_NAMES = ("xpt_R", "xpt_Z")


def _coerce_named(
    geometry,
    names: Sequence[str] | None,
) -> dict[str, float]:
    """Normalise a geometry input to a ``{component_name: value}`` mapping.

    Accepts either a mapping already keyed by component name, or an
    ``EquilibriumGeometry``-like object (``.target`` ``(F>=1, D)`` + ``.names``
    + optional ``.finite_mask``) — in which case the FIRST frame is used and
    masked / non-finite components are dropped.
    """
    import numpy as np  # noqa: PLC0415

    if isinstance(geometry, Mapping):
        out: dict[str, float] = {}
        for k, v in geometry.items():
            fv = float(v)
            if math.isfinite(fv):
                out[str(k)] = fv
        return out

    target = np.asarray(geometry.target, dtype=np.float64)
    if target.ndim == 1:
        row = target
        mask_row = None
        if getattr(geometry, "finite_mask", None) is not None:
            mask_row = np.asarray(geometry.finite_mask, dtype=bool)
    else:
        row = target[0]
        mask_row = None
        if getattr(geometry, "finite_mask", None) is not None:
            mask_row = np.asarray(geometry.finite_mask, dtype=bool)[0]
    comp_names = names if names is not None else getattr(geometry, "names", None)
    if comp_names is None:
        raise ValueError("geometry has no component names; pass names=")
    out = {}
    for i, name in enumerate(comp_names):
        val = float(row[i])
        finite = math.isfinite(val) and (mask_row is None or bool(mask_row[i]))
        if finite:
            out[str(name)] = val
    return out


def _pair_distance(
    cand: dict[str, float], ref: dict[str, float], names: tuple[str, str]
) -> float | None:
    """Euclidean distance between a candidate / reference 2-component pair."""
    if not all(n in cand and n in ref for n in names):
        return None
    dx = cand[names[0]] - ref[names[0]]
    dy = cand[names[1]] - ref[names[1]]
    return math.hypot(dx, dy)


def judge_geometry(
    candidate,
    reference,
    *,
    names: Sequence[str] | None = None,
    axis_tol: float = 0.02,
    xpoint_tol: float = 0.03,
    boundary_tol: float = 0.03,
) -> GeometryVerdict:
    """Score a model-produced geometry against the EFIT reference.

    Both ``candidate`` and ``reference`` may be a ``{component_name: value}``
    mapping or an ``EquilibriumGeometry``-like object (first frame used, masked
    components dropped).  Component names follow
    :mod:`imas_ambix.worldmodel.equilibrium_labels` (``axis_R, axis_Z, xpt_R,
    xpt_Z, lcfs_r_0..7``).

    A perfect match yields ~0 error and ``passed=True``; an offset yields the
    expected Euclidean / RMS error and ``passed`` reflects whether every
    *available* error is within tolerance.  Components masked in the reference
    or absent from the candidate are reported as unavailable (``None`` /
    excluded), never silently scored as zero.

    Parameters
    ----------
    candidate, reference:
        Geometry to compare (model output vs. EFIT reconstruction).
    names:
        Component-name override when passing bare ``EquilibriumGeometry``-like
        objects lacking ``.names``; ignored for mappings.
    axis_tol, xpoint_tol, boundary_tol:
        Pass tolerances in metres for the axis Euclidean error, X-point
        Euclidean error, and LCFS-radii RMS respectively.

    Returns
    -------
    GeometryVerdict
    """
    cand = _coerce_named(candidate, names)
    ref = _coerce_named(reference, names)

    verdict = GeometryVerdict()

    # Axis and X-point Euclidean errors.
    verdict.axis_error = _pair_distance(cand, ref, _AXIS_NAMES)
    verdict.xpoint_error = _pair_distance(cand, ref, _XPOINT_NAMES)
    if verdict.axis_error is None:
        verdict.notes.append("axis: component(s) masked or missing")
    if verdict.xpoint_error is None:
        verdict.notes.append("x-point: component(s) masked or missing")

    # Boundary: RMS over LCFS control-point radii present in both.
    sq = 0.0
    n_bdy = 0
    for name in (n for n in ref if n.startswith("lcfs_r")):
        if name in cand:
            err = cand[name] - ref[name]
            verdict.components[name] = abs(err)
            sq += err * err
            n_bdy += 1
    verdict.n_boundary_points = n_bdy
    if n_bdy > 0:
        verdict.boundary_rms = math.sqrt(sq / n_bdy)

    # Per-component absolute errors for axis / X-point too.
    for name in (*_AXIS_NAMES, *_XPOINT_NAMES):
        if name in cand and name in ref:
            verdict.components[name] = abs(cand[name] - ref[name])

    # Pass verdict: every AVAILABLE error within tolerance; None if nothing
    # was comparable at all.
    checks: list[bool] = []
    if verdict.axis_error is not None:
        checks.append(verdict.axis_error <= axis_tol)
    if verdict.xpoint_error is not None:
        checks.append(verdict.xpoint_error <= xpoint_tol)
    if verdict.boundary_rms is not None:
        checks.append(verdict.boundary_rms <= boundary_tol)
    verdict.passed = all(checks) if checks else None
    if not checks:
        verdict.notes.append("no comparable component: verdict undefined")

    return verdict


__all__ = [
    "FirewallViolation",
    "evaluator_context",
    "read_efit_geometry",
    "GeometryVerdict",
    "judge_geometry",
]
