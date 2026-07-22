"""Incomplete elliptic integral of the third kind, Π(n; φ, m).

    Π(n; φ, m) = ∫₀^φ dθ / [(1 − n sin²θ) √(1 − m sin²θ)]        (m = k²)

is the one special function the Urankar Part V polygon-section field formulae
need that :mod:`scipy.special` does not ship (scipy has the complete Π only
implicitly, and the incomplete first/second kinds).  It is built here on scipy's
Carlson symmetric forms ``elliprf`` (R_F) and ``elliprj`` (R_J), which are
themselves symmetric, uniformly convergent by duplication, and — for the
characteristic ``n > 1`` — return the Cauchy principal value, so no separate
branch handling is needed for the pole the polygon integrals cross.

Carlson representation (DLMF 19.25.14), with s = sin φ, c = cos φ:

    Π(n; φ, m) = s·R_F(c², 1 − m s², 1) + (n/3) s³·R_J(c², 1 − m s², 1, 1 − n s²)

valid for |φ| ≤ π/2.  General real amplitude is reached by the half-period
reduction Π(n; φ + jπ, m) = 2j·Π(n; m) + Π(n; φ_red, m), where Π(n; m) is the
complete integral (the φ = π/2 limit) and φ_red ∈ [−π/2, π/2].

The complete Π reduces to ``cylinder._ellipp``; this module is the incomplete
generalisation the polygon kernel rests on.
"""

from __future__ import annotations

import numpy as np
import scipy.special  # type: ignore[import-untyped]

__all__ = ["ellippi", "ellippi_complete"]


def ellippi_complete(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Complete elliptic integral of the third kind Π(n; m) = Π(n; π/2, m).

    Carlson form at φ = π/2 (s = 1, c = 0): R_F(0, 1−m, 1) + (n/3) R_J(0, 1−m, 1, 1−n).
    """
    n = np.asarray(n, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    zero = np.zeros(np.broadcast(n, m).shape)
    y = 1.0 - m + zero
    z = np.ones_like(y)
    x = zero
    p = 1.0 - n + zero
    rf = scipy.special.elliprf(x, y, z)
    rj = scipy.special.elliprj(x, y, z, p)
    return rf + rj * (n + zero) / 3.0


def ellippi(n: np.ndarray, phi: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Incomplete elliptic integral of the third kind Π(n; φ, m), m = k².

    Vectorised over all three arguments (broadcasting).  Handles arbitrary real
    amplitude φ via half-period reduction and the ``n > 1`` Cauchy-principal-value
    branch via Carlson R_J with a negative fourth argument.
    """
    n, phi, m = (np.asarray(v, dtype=np.float64) for v in (n, phi, m))
    n, phi, m = np.broadcast_arrays(n, phi, m)

    # half-period reduction: Π(n; φ + jπ, m) = 2j·Π_complete + Π(n; φ_red, m),
    # φ_red ∈ [−π/2, π/2] where the single-argument Carlson form is valid.
    j = np.round(phi / np.pi)
    phir = phi - j * np.pi

    s = np.sin(phir)
    c = np.cos(phir)
    s2 = s * s
    x = c * c
    y = 1.0 - m * s2
    z = np.ones_like(x)
    p = 1.0 - n * s2
    rf = scipy.special.elliprf(x, y, z)
    rj = scipy.special.elliprj(x, y, z, p)
    base = s * rf + n * s2 * s / 3.0 * rj

    out = base
    if np.any(j != 0.0):
        out = base + 2.0 * j * ellippi_complete(n, m)
    return out
