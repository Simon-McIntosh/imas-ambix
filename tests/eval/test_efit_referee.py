"""Tests for the firewalled EFIT referee.

These prove the boundary guard the referee exists to enforce:

(a) the referee reads/returns EFIT geometry inside an evaluator context
    (driven on a synthetic loader, so no /work access is needed);
(b) the same read OUTSIDE the evaluator context raises ``FirewallViolation``;
(c) a static import guard — the world-model training source tree does not
    reference ``efit_referee`` at all (proof it is off the training path);
(d) the judge scores a known geometry pair correctly (match → ~0; offset →
    the expected error).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.eval.efit_referee import (
    FirewallViolation,
    GeometryVerdict,
    evaluator_context,
    judge_geometry,
    read_efit_geometry,
)

# Component layout (axis_R, axis_Z, xpt_R, xpt_Z, lcfs_r_0..7) — kept local so
# the test does not import the world-model package (which would defeat the
# static-import guard in (c)).
_NAMES = (
    "axis_R",
    "axis_Z",
    "xpt_R",
    "xpt_Z",
    *tuple(f"lcfs_r_{k}" for k in range(8)),
)


class _StubGeometry:
    """Minimal EquilibriumGeometry-like object for the injected loader."""

    def __init__(self, target: np.ndarray, names=_NAMES, finite_mask=None):
        self.target = np.asarray(target, dtype=np.float32)
        self.names = names
        self.finite_mask = (
            np.isfinite(self.target) if finite_mask is None else finite_mask
        )
        self.shot_id = 12345
        self.units = "m"


def _stub_loader(shot_id, frame_times, **kwargs):
    """Synthetic EFIT geometry — one frame, all 12 components finite."""
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    row = np.array(
        [0.90, 0.01, 0.55, -1.10, *(0.60 + 0.01 * np.arange(8))],
        dtype=np.float32,
    )
    target = np.tile(row, (max(ft.size, 1), 1))
    return _StubGeometry(target)


# --- (a) reads inside an evaluator context ---------------------------------


def test_reads_inside_evaluator_context():
    ft = np.array([0.30, 0.31, 0.32])
    with evaluator_context():
        geom = read_efit_geometry(99999, ft, loader=_stub_loader)
    assert geom.target.shape == (3, 12)
    assert geom.names == _NAMES
    # The axis components survive the round-trip.
    assert geom.target[0, 0] == pytest.approx(0.90)
    assert geom.target[0, 1] == pytest.approx(0.01)


def test_evaluator_context_is_reentrant_and_closes():
    ft = np.array([0.30])
    with evaluator_context():
        with evaluator_context():  # nested
            read_efit_geometry(1, ft, loader=_stub_loader)
        # still open after the inner context exits
        read_efit_geometry(1, ft, loader=_stub_loader)
    # closed again outside
    with pytest.raises(FirewallViolation):
        read_efit_geometry(1, ft, loader=_stub_loader)


# --- (b) read OUTSIDE the context raises -----------------------------------


def test_read_outside_context_raises():
    ft = np.array([0.30])
    with pytest.raises(FirewallViolation):
        read_efit_geometry(1, ft, loader=_stub_loader)


def test_gate_closes_even_after_exception():
    ft = np.array([0.30])

    def _boom(shot_id, frame_times, **kwargs):
        raise ValueError("loader blew up")

    with pytest.raises(ValueError), evaluator_context():
        read_efit_geometry(1, ft, loader=_boom)
    # the gate must be closed again after the body raised
    with pytest.raises(FirewallViolation):
        read_efit_geometry(1, ft, loader=_stub_loader)


# --- (c) static import guard: training tree never references the referee ----


def test_training_tree_does_not_reference_referee():
    """No world-model / training source references ``efit_referee``.

    This is the proof the referee is off the training input path: a leak would
    show up as an import or symbol reference in the model / training tree.
    """
    repo_root = Path(__file__).resolve().parents[2]
    # Directories that hold model definitions and training loops.
    search_roots = [
        repo_root / "imas_ambix" / "worldmodel",
        repo_root / "imas_ambix" / "tokenizer",
        repo_root / "imas_ambix" / "statespace",
    ]
    pattern = re.compile(r"efit_referee")
    hits: list[str] = []
    for root in search_roots:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{py}:{lineno}: {line.strip()}")
    assert not hits, "training/model tree references efit_referee:\n" + "\n".join(hits)


def test_referee_does_not_import_worldmodel_at_module_load():
    """Importing the referee must not pull in the world-model package.

    The default loader imports ``equilibrium_labels`` lazily; at module-load
    time there must be no top-level world-model coupling.  We import the module
    fresh and confirm no ``imas_ambix.worldmodel`` submodule was dragged in by
    that import alone.
    """
    import importlib
    import sys

    # Drop any cached worldmodel submodules so we measure this import only.
    for mod in list(sys.modules):
        if mod.startswith("imas_ambix.worldmodel"):
            del sys.modules[mod]
    sys.modules.pop("imas_ambix.eval.efit_referee", None)

    importlib.import_module("imas_ambix.eval.efit_referee")

    leaked = [m for m in sys.modules if m.startswith("imas_ambix.worldmodel")]
    assert not leaked, f"referee import leaked world-model modules: {leaked}"


# --- (d) judge scores a known pair correctly --------------------------------


def _ref_dict():
    return {
        "axis_R": 0.90,
        "axis_Z": 0.01,
        "xpt_R": 0.55,
        "xpt_Z": -1.10,
        **{f"lcfs_r_{k}": 0.60 + 0.01 * k for k in range(8)},
    }


def test_judge_perfect_match_is_zero_and_passes():
    ref = _ref_dict()
    cand = dict(ref)
    v = judge_geometry(cand, ref)
    assert isinstance(v, GeometryVerdict)
    assert v.axis_error == pytest.approx(0.0)
    assert v.xpoint_error == pytest.approx(0.0)
    assert v.boundary_rms == pytest.approx(0.0)
    assert v.n_boundary_points == 8
    assert v.passed is True


def test_judge_offset_gives_expected_error():
    ref = _ref_dict()
    cand = dict(ref)
    # Shift the axis by (0.03, 0.04) -> Euclidean 0.05 m (3-4-5 triangle).
    cand["axis_R"] += 0.03
    cand["axis_Z"] += 0.04
    # Shift the X-point by 0.10 m in R only.
    cand["xpt_R"] += 0.10
    # Offset every boundary radius by a constant 0.05 m -> RMS 0.05.
    for k in range(8):
        cand[f"lcfs_r_{k}"] += 0.05
    v = judge_geometry(cand, ref)
    assert v.axis_error == pytest.approx(0.05)
    assert v.xpoint_error == pytest.approx(0.10)
    assert v.boundary_rms == pytest.approx(0.05)
    # axis (0.05 > 0.02) and x-point (0.10 > 0.03) both exceed tolerance.
    assert v.passed is False


def test_judge_accepts_equilibriumgeometry_like_objects():
    """The judge takes EquilibriumGeometry-like objects, not just dicts."""
    ref_row = np.array(
        [0.90, 0.01, 0.55, -1.10, *(0.60 + 0.01 * np.arange(8))],
        dtype=np.float32,
    )
    cand_row = ref_row.copy()
    cand_row[0] += 0.03
    cand_row[1] += 0.04
    ref = _StubGeometry(np.tile(ref_row, (2, 1)))
    cand = _StubGeometry(np.tile(cand_row, (2, 1)))
    v = judge_geometry(cand, ref)
    assert v.axis_error == pytest.approx(0.05, abs=1e-5)
    assert v.boundary_rms == pytest.approx(0.0, abs=1e-5)


def test_judge_masked_reference_component_is_unavailable_not_zero():
    """A masked reference X-point yields ``None`` error, never a false zero."""
    ref_row = np.array(
        [0.90, 0.01, np.nan, np.nan, *(0.60 + 0.01 * np.arange(8))],
        dtype=np.float32,
    )
    cand_row = np.array(
        [0.90, 0.01, 0.55, -1.10, *(0.60 + 0.01 * np.arange(8))],
        dtype=np.float32,
    )
    ref = _StubGeometry(ref_row)
    cand = _StubGeometry(cand_row)
    v = judge_geometry(cand, ref)
    assert v.axis_error == pytest.approx(0.0)
    assert v.xpoint_error is None  # masked in reference -> unavailable
    assert v.boundary_rms == pytest.approx(0.0)


def test_judge_no_comparable_component_is_undefined():
    v = judge_geometry({"foo": 1.0}, {"bar": 2.0})
    assert v.passed is None
    assert v.axis_error is None
    assert v.xpoint_error is None
    assert v.boundary_rms is None
