"""Pin the process thread-pool environment variables to a fixed count.

A BLAS or OpenMP library reads its thread-count variable once, when it loads, so
the value must be in the environment before numpy, scikit-learn or torch is
imported.  :func:`pin_thread_pools` writes the three variables a numerical stack
consults --- ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and
``MKL_NUM_THREADS`` --- each with :func:`os.environ.setdefault`, so a value an
outer caller has already chosen wins.

This module is deliberately stdlib-only: importing it must not pull in a BLAS
library, or the pin would land too late.  Both callers share it instead of
keeping a copy of the loop --- a compute-only probe that pins its own pools, and
the test session's per-worker cap.
"""

from __future__ import annotations

import os

# The thread-count variables a BLAS/OpenMP library consults at load time.
POOL_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_thread_pools(count: int) -> None:
    """Set the BLAS/OpenMP thread-pool variables to ``count`` if unset.

    Each variable is written with :func:`os.environ.setdefault`: an explicit
    outer setting is preserved, and only absent variables take ``count``.
    """
    for var in POOL_ENV_VARS:
        os.environ.setdefault(var, str(count))
