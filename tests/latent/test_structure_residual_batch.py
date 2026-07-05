"""Numerical-equivalence CI test for the examples-batched structure residual.

Non-negotiable safety net (per the scope-extension ruling on
imas_ambix/latent/structure_residual.py): a subtly-wrong physics residual is
worse than a slow one.  This pins :func:`structure_residual_batch` to the
EXACT output (forward AND gradients) of calling the per-example
:func:`structure_residual` once per row, across many random configurations
including the ``connectivity="locality"`` path (the one that broke
``torch.vmap`` and motivated writing a real batched form instead).
"""

from __future__ import annotations

import torch

from imas_ambix.latent.structure_residual import (
    structure_residual,
    structure_residual_batch,
)

REL_TOL = 1e-6
ABS_TOL = 1e-9


def _random_geometry(rng: torch.Generator, n: int, dtype: torch.dtype):
    r_c = torch.rand(n, dtype=dtype, generator=rng) + 0.5
    z_c = torch.randn(n, dtype=dtype, generator=rng) * 0.4
    return r_c, z_c


def test_forward_matches_per_example_loop_across_configs():
    rng = torch.Generator().manual_seed(0)
    dtype = torch.float64
    for trial in range(20):
        e = int(torch.randint(2, 12, (1,), generator=rng).item())
        n = int(torch.randint(20, 300, (1,), generator=rng).item())
        psi_c = torch.randn(e, n, dtype=dtype, generator=rng) * (1 + trial * 0.3)
        r_c, z_c = _random_geometry(rng, n, dtype)
        jphi_c = torch.randn(e, n, dtype=dtype, generator=rng) * (10.0 ** (trial % 5))
        if trial % 7 == 0:  # exercise the current-free (valid=False) row
            jphi_c[0] = 0.0

        for connectivity in (None, "locality"):
            for form in ("affine-r2", "jphi"):
                batched = structure_residual_batch(
                    psi_c, r_c, jphi_c, z_c=z_c, connectivity=connectivity, form=form
                )
                looped = torch.stack(
                    [
                        structure_residual(
                            psi_c[k],
                            r_c,
                            jphi_c[k],
                            z_c=z_c,
                            connectivity=connectivity,
                            form=form,
                        )
                        for k in range(e)
                    ]
                )
                assert torch.allclose(batched, looped, atol=ABS_TOL, rtol=REL_TOL), (
                    f"trial={trial} e={e} n={n} connectivity={connectivity} "
                    f"form={form}: max diff "
                    f"{(batched - looped).abs().max().item():.3e}"
                )


def test_gradients_match_per_example_loop():
    """The residual is differentiated through in engine.py -- pin gradients,
    not just the forward value, for both connectivity arms."""
    torch.manual_seed(1)
    dtype = torch.float64
    e, n = 5, 80
    r_c = torch.rand(n, dtype=dtype) + 0.5
    z_c = torch.randn(n, dtype=dtype) * 0.4

    for connectivity in (None, "locality"):
        psi_c = torch.randn(e, n, dtype=dtype, requires_grad=True)
        jphi_c = (torch.randn(e, n, dtype=dtype) * 1e4).requires_grad_(True)

        out_batch = structure_residual_batch(
            psi_c, r_c, jphi_c, z_c=z_c, connectivity=connectivity
        )
        out_batch.sum().backward()
        grad_psi_batch = psi_c.grad.clone()
        grad_jphi_batch = jphi_c.grad.clone()

        psi_c2 = psi_c.detach().clone().requires_grad_(True)
        jphi_c2 = jphi_c.detach().clone().requires_grad_(True)
        out_loop = torch.stack(
            [
                structure_residual(
                    psi_c2[k], r_c, jphi_c2[k], z_c=z_c, connectivity=connectivity
                )
                for k in range(e)
            ]
        )
        out_loop.sum().backward()

        assert torch.allclose(out_batch, out_loop, atol=ABS_TOL, rtol=REL_TOL)
        assert torch.allclose(grad_psi_batch, psi_c2.grad, atol=1e-6, rtol=1e-4), (
            f"connectivity={connectivity}: psi grad mismatch"
        )
        assert torch.allclose(grad_jphi_batch, jphi_c2.grad, atol=1e-6, rtol=1e-4), (
            f"connectivity={connectivity}: jphi grad mismatch"
        )


def test_all_examples_current_free_returns_zero():
    """Every example has zero current -- the batched form's fixed-shape
    ``valid`` mask must zero the whole output, matching the per-example
    early-return path, with no NaN/Inf leaking through the solve."""
    dtype = torch.float64
    e, n = 4, 30
    r_c = torch.rand(n, dtype=dtype) + 0.5
    z_c = torch.randn(n, dtype=dtype) * 0.4
    psi_c = torch.randn(e, n, dtype=dtype)
    jphi_c = torch.zeros(e, n, dtype=dtype)

    out = structure_residual_batch(psi_c, r_c, jphi_c, z_c=z_c, connectivity="locality")
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros(e, dtype=dtype))


def test_labels_connectivity_rejected():
    dtype = torch.float64
    e, n = 3, 20
    psi_c = torch.randn(e, n, dtype=dtype)
    r_c = torch.rand(n, dtype=dtype) + 0.5
    jphi_c = torch.randn(e, n, dtype=dtype)
    try:
        structure_residual_batch(psi_c, r_c, jphi_c, connectivity="labels")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for connectivity='labels'")
