"""Fixtures shared by the agent test suite.

The router gate's control file now defaults to the group-owned control directory
under the project base, ``<base_dir>/agents/control/router-gate.json``. That is
the right production default and the wrong thing for a test to resolve: it is a
real path on shared storage, so a test that resolved a control file with no
override would read and write production control state.

The fixture below therefore drives the site control path from its own
environment variable for every test. It sets that variable to the **empty
string** rather than to a scratch file, and the difference matters. An empty
site value suppresses the site branch, leaving a test that names nothing on the
lane document's sibling -- which is the contract the resolution tests pin and
which a scratch-file value would displace, sending those tests to a path that is
neither the sibling nor the production file. A test that is about the site
default sets the variable itself; either way no test can resolve the production
path.
"""

from __future__ import annotations

import os

import pytest

from imas_ambix._thread_pools import pin_thread_pools
from imas_ambix.agent.profile import GATE_CONTROL_PATH_ENV


@pytest.fixture(autouse=True)
def _no_production_gate_control_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test off the production gate control file."""
    monkeypatch.setenv(GATE_CONTROL_PATH_ENV, "")


# ---------------------------------------------------------------------------
# Cap each xdist worker's thread pools at its share of the node's CPUs
# ---------------------------------------------------------------------------
#
# A whole-suite run at ``-n 16`` on a 20-CPU node drove every worker's torch and
# BLAS pools to one thread per core --- about 320 threads over 20 cores --- so a
# test that measures two minutes alone runs several times slower under the suite
# and trips the 300 s per-test bound.  Each worker instead pins its pools to its
# share of the CPUs available to the process.
#
# This is the same pin already applied once in
# ``imas_ambix/statespace/oracle_probe.py``, which sets the BLAS/OpenMP
# variables with ``setdefault`` before importing numpy, because a BLAS library
# reads them only when it loads; both callers now share it through
# ``imas_ambix/_thread_pools.py``.  A test session cannot rely on import
# ordering alone, since a pytest plugin may load numpy before this conftest
# runs, so the cap also limits already-loaded BLAS pools through threadpoolctl
# and sets torch's pool explicitly.  Without xdist it changes nothing.


def available_cpus() -> int:
    """CPUs this process may run on, honouring CPU affinity where present."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def worker_share(available: int, worker_count: int) -> int:
    """One xdist worker's share of ``available`` CPUs, at least one."""
    return max(1, available // worker_count)


# A live threadpool_limits context manager must be held, not merely entered: if
# it is garbage-collected its __exit__ restores the original pool sizes.
_session_pool_limiter = None


def _limit_loaded_pools(share: int) -> None:
    """Cap already-loaded BLAS pools for the rest of the session.

    ``threadpool_limits`` is scoped to the ``blas`` API deliberately.  Capping
    the ``openmp`` API would also set the OpenMP runtime that torch reports
    through ``torch.get_num_threads``, which would mask the explicit
    ``torch.set_num_threads`` below and leave this cap's torch branch untested.
    OpenMP pools not yet loaded take the share from the environment pin; a pool
    loaded by a plugin before this conftest ran is the residual case the BLAS
    limit does not reach.
    """
    global _session_pool_limiter

    from threadpoolctl import threadpool_limits

    _session_pool_limiter = threadpool_limits(limits=share, user_api="blas")
    _session_pool_limiter.__enter__()


def _resolve_torch():
    """Return the torch module, or None when it is not importable."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def apply_worker_thread_cap(
    worker_count: str | None = None, available: int | None = None
) -> int | None:
    """Cap this worker's thread pools; return the share, or None without xdist."""
    if worker_count is None:
        worker_count = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if not worker_count:
        return None
    if available is None:
        available = available_cpus()
    share = worker_share(available, int(worker_count))
    # Pin the environment before torch is resolved: importing torch loads its
    # OpenMP and MKL runtimes, which read the thread variables once at load
    # time, so the pin must land first or torch's own pools start at one
    # thread per core.  torch.set_num_threads then states the count explicitly
    # rather than relying on the load-time value alone.
    pin_thread_pools(share)
    _limit_loaded_pools(share)
    torch = _resolve_torch()
    if torch is not None:
        torch.set_num_threads(share)
    return share


apply_worker_thread_cap()
