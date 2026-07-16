"""Split scorer robustness — the operator arm must stay finite for any operator.

The p′/FF′ split score re-expresses the operator's Ip-normalised correction
``dc`` onto the raw ladder coefficients and reads the p′-group current
fraction.  A correction large enough to drive every non-negative profile DOF
to zero collapses the coefficient vector to all-zeros, at which point the bare
``split_fraction`` is undefined (zero total current).  Left unguarded, a single
such slice makes ``np.median`` over the sequence NaN — the failure that
appeared with an under-regularised checkpoint whose corrections saturated at
±dc_scale.  These pin that the robust helper returns a FINITE, honest failure
score for the degenerate case while reproducing ``split_fraction`` exactly when
the corrected profile is valid.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.gs_solve import profile_basis
from scripts.synthetic_eddy_pretrain import operator_split_fraction, split_fraction


def _synthetic_slice(seed: int = 0, n: int = 200):
    """A realistic per-cell ψ_N map, radii and non-negative ladder coeffs."""
    rng = np.random.default_rng(seed)
    psi_n = np.clip(rng.uniform(0.0, 1.0, n), 0.0, 1.5)
    r_cells = rng.uniform(0.3, 1.4, n)
    r0 = 0.9
    c_fit = rng.uniform(0.2, 1.0, 6)
    images = profile_basis(psi_n, r_cells, r0=r0, n_p=3, n_f=3, kind="monomial-nonneg")
    gross_raw = np.abs(images).sum(axis=0).clip(min=1e-30)
    # non-degenerate columns carry the full Ip after profile_columns rescaling
    ip = 6.0e5
    gross_col = np.where(gross_raw > 1e-20, ip, 0.0)
    return psi_n, r_cells, r0, c_fit, gross_col, gross_raw


def test_bare_split_fraction_nan_on_zero_current():
    """The documented degenerate: an all-zero coefficient vector has no split."""
    psi_n, r_cells, r0, *_ = _synthetic_slice()
    assert not np.isfinite(split_fraction(np.zeros(6), psi_n, r_cells, r0))
    # a valid non-negative profile is finite and in [0, 1]
    s = split_fraction(np.full(6, 0.5), psi_n, r_cells, r0)
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_saturated_negative_correction_collapses_but_scores_finite():
    """A saturated negative dc annihilates every DOF — must still score finite."""
    psi_n, r_cells, r0, c_fit, gross_col, gross_raw = _synthetic_slice()
    s_true = split_fraction(c_fit, psi_n, r_cells, r0)
    assert np.isfinite(s_true)

    # this correction drives c_op to all-zeros under the current re-expression
    dc = -0.3 * np.ones(6)
    c_op = np.clip(c_fit + dc * gross_col / gross_raw, 0.0, None)
    assert c_op.sum() == 0.0  # the collapse the bare scorer cannot read
    assert not np.isfinite(split_fraction(c_op, psi_n, r_cells, r0))

    # the robust helper flags the collapse and scores it as an operator failure
    s_op, degenerate = operator_split_fraction(
        c_fit, dc, gross_col, gross_raw, psi_n, r_cells, r0, s_true
    )
    assert degenerate is True
    assert np.isfinite(s_op)
    # worst attainable fraction error: >= 0.5, well past the ~0.1 spine baseline
    assert abs(s_op - s_true) >= 0.5


def test_mild_correction_matches_bare_split_fraction():
    """When the profile stays valid, the helper equals the bare scorer exactly."""
    psi_n, r_cells, r0, c_fit, gross_col, gross_raw = _synthetic_slice()
    s_true = split_fraction(c_fit, psi_n, r_cells, r0)
    dc = np.array([0.1, -0.05, 0.02, -0.03, 0.04, -0.01])
    c_op = np.clip(c_fit + dc * gross_col / gross_raw, 0.0, None)
    s_ref = split_fraction(c_op, psi_n, r_cells, r0)
    assert np.isfinite(s_ref)

    s_op, degenerate = operator_split_fraction(
        c_fit, dc, gross_col, gross_raw, psi_n, r_cells, r0, s_true
    )
    assert degenerate is False
    assert s_op == s_ref


def test_median_stays_finite_with_one_degenerate_slice():
    """One collapsed slice must not poison the aggregate over a sequence."""
    errs, n_degenerate = [], 0
    for seed in range(8):
        psi_n, r_cells, r0, c_fit, gross_col, gross_raw = _synthetic_slice(seed)
        s_true = split_fraction(c_fit, psi_n, r_cells, r0)
        # slice 3 gets the saturated collapse; the rest a mild valid correction
        dc = (-0.3 * np.ones(6)) if seed == 3 else np.full(6, 0.02)
        s_op, degenerate = operator_split_fraction(
            c_fit, dc, gross_col, gross_raw, psi_n, r_cells, r0, s_true
        )
        n_degenerate += int(degenerate)
        errs.append(abs(s_op - s_true))
    assert n_degenerate == 1
    assert np.isfinite(np.median(errs))
    assert np.isfinite(np.mean(errs))
