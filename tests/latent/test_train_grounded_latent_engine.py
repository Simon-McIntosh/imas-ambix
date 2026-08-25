"""Tests for the two training-loop safety mechanisms added after job
1225447's post-mortem (an undetached closure aux loss drove ``i_cell`` to
60-2800x its expected scale over ~18000 unmonitored steps):

* :func:`_check_i_cell_scale` — a hard, per-step assert on the per-cell
  current's magnitude relative to ``Ip / n_cells``.
* :func:`_maybe_update_disc` — throttles :class:`DiscrepancyLambda`'s
  post-warm-up λ ratchet to once every ``adapt_every`` steps, matching the
  locked per-slice variational-inverse policy's cadence
  (:class:`imas_ambix.latent.patch_inverse.InverseConfig.adapt_every`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.train_grounded_latent_engine as tgle
from imas_ambix.latent.patch_encoder import DiscrepancyLambda


def test_check_i_cell_scale_passes_for_normal_current():
    n_cells = 100
    ip = torch.full((4,), 1.0e5, dtype=torch.float64)
    i_cell = torch.full((4, n_cells), 1.0e5 / n_cells, dtype=torch.float64)
    tgle._check_i_cell_scale(i_cell, ip, max_ratio=50.0)  # must not raise


def test_check_i_cell_scale_raises_for_divergent_current():
    n_cells = 100
    ip = torch.full((4,), 1.0e5, dtype=torch.float64)
    i_cell = torch.full((4, n_cells), 1.0e5 / n_cells, dtype=torch.float64)
    i_cell[2] *= 200.0  # one example over threshold — job 1225447's signature
    with pytest.raises(RuntimeError, match="i_cell scale sanity check failed"):
        tgle._check_i_cell_scale(i_cell, ip, max_ratio=50.0)


def test_check_i_cell_scale_zero_current_never_divides_by_zero():
    """A quiescent/vacuum slice (Ip ~ 0) must not spuriously trip the guard
    -- the expected scale floors at 1.0 A, not zero."""
    n_cells = 50
    ip = torch.zeros(3, dtype=torch.float64)
    i_cell = torch.zeros(3, n_cells, dtype=torch.float64)
    tgle._check_i_cell_scale(i_cell, ip, max_ratio=50.0)  # must not raise


def test_maybe_update_disc_throttles_ratchet_to_adapt_every():
    """Post warm-up, DiscrepancyLambda.update ratchets λ on every call --
    _maybe_update_disc must only let one in every `adapt_every` calls
    through, matching the locked per-slice policy's cadence."""
    disc = DiscrepancyLambda(
        n_examples=4,
        warmup_epochs=1,
        lam0=3.0,
        ratio=1.5,
        adapt_factor=1.5,
        lam_max=1e6,
        device="cpu",
        dtype=torch.float64,
    )
    ids = np.arange(4)
    warm_misfit = torch.full((4,), 2.0, dtype=torch.float64)
    # Below the frozen target, so the discrepancy multiplier increases.
    low_misfit = torch.full((4,), 1.0, dtype=torch.float64)

    # epoch 0 (step 0): record the warm-up misfit
    tgle._maybe_update_disc(disc, ids, warm_misfit, epoch=0, step=0, adapt_every=5)
    # epoch 1 == warmup_epochs (step 1): freeze target = ratio * warm_misfit = 3.0
    tgle._maybe_update_disc(disc, ids, warm_misfit, epoch=1, step=1, adapt_every=5)
    lam_after_freeze = disc.lam.clone()

    adapt_every = 5
    for step in range(2, 27):
        tgle._maybe_update_disc(
            disc, ids, low_misfit, epoch=2, step=step, adapt_every=adapt_every
        )

    expected_ratchets = sum(1 for step in range(2, 27) if step % adapt_every == 0)
    assert expected_ratchets == 5  # steps 5, 10, 15, 20, 25
    expected_lam = lam_after_freeze * (1.5**expected_ratchets)
    assert torch.allclose(disc.lam, expected_lam)


def test_maybe_update_disc_ratchets_faster_when_untethered():
    """Sanity check on the throttle's direction: calling every step (as the
    shared class does with no throttle) must ratchet strictly faster than
    calling once every `adapt_every` steps over the same span."""
    kwargs = dict(
        n_examples=4,
        warmup_epochs=1,
        lam0=3.0,
        ratio=1.5,
        adapt_factor=1.5,
        lam_max=1e6,
        device="cpu",
        dtype=torch.float64,
    )
    ids = np.arange(4)
    warm_misfit = torch.full((4,), 2.0, dtype=torch.float64)
    # Below the frozen target, so the discrepancy multiplier increases.
    low_misfit = torch.full((4,), 1.0, dtype=torch.float64)

    disc_throttled = DiscrepancyLambda(**kwargs)
    disc_untethered = DiscrepancyLambda(**kwargs)
    for disc in (disc_throttled, disc_untethered):
        tgle._maybe_update_disc(disc, ids, warm_misfit, epoch=0, step=0, adapt_every=25)
        tgle._maybe_update_disc(disc, ids, warm_misfit, epoch=1, step=1, adapt_every=25)

    for step in range(2, 27):
        tgle._maybe_update_disc(
            disc_throttled, ids, low_misfit, epoch=2, step=step, adapt_every=25
        )
        # adapt_every=1 -- every call passes the throttle, matching the
        # shared class's own (un-throttled) per-call ratchet
        tgle._maybe_update_disc(
            disc_untethered, ids, low_misfit, epoch=2, step=step, adapt_every=1
        )

    assert float(disc_untethered.lam[0]) > float(disc_throttled.lam[0])


def test_maybe_update_disc_never_throttles_during_warmup():
    """Warm-up recording (epoch <= warmup_epochs) must go through on every
    call regardless of `adapt_every` -- only the post-warm-up ratchet is
    throttled."""
    disc = DiscrepancyLambda(
        n_examples=5, warmup_epochs=2, device="cpu", dtype=torch.float64
    )
    for step in range(5):
        ids = np.array([step])
        misfit = torch.tensor([float(step) + 1.0], dtype=torch.float64)
        tgle._maybe_update_disc(disc, ids, misfit, epoch=0, step=step, adapt_every=100)
    assert torch.allclose(
        disc._warm_misfit, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    )
