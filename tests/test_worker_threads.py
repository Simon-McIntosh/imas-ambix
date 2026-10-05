"""The per-worker thread-pool cap set by :mod:`tests.conftest`.

A whole-suite xdist run oversubscribes the node: every worker's torch and BLAS
pools default to one thread per core, so a test that takes minutes alone runs
several times slower and trips the per-test bound.  The conftest caps each
worker at its share of the CPUs available to the process.  These tests pin the
cap's pieces directly --- the shared environment pin, the share computation, the
no-xdist no-op, and, when the suite itself runs under xdist, that torch's pool
actually took the share.
"""

from __future__ import annotations

import os

import pytest

from imas_ambix._thread_pools import POOL_ENV_VARS, pin_thread_pools
from tests import conftest


@pytest.fixture()
def _clear_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start a test from a known-absent pool environment."""
    for var in POOL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_pin_thread_pools_sets_the_three_variables(
    _clear_pool_env: None,
) -> None:
    pin_thread_pools(3)
    assert [os.environ[var] for var in POOL_ENV_VARS] == ["3", "3", "3"]


def test_pin_thread_pools_leaves_a_preset_variable_unchanged(
    monkeypatch: pytest.MonkeyPatch, _clear_pool_env: None
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    pin_thread_pools(3)
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"


@pytest.mark.parametrize(
    ("available", "worker_count", "expected"),
    [
        (8, 4, 2),
        (20, 16, 1),
        (64, 4, 16),
        (3, 8, 1),
        (1, 1, 1),
    ],
)
def test_worker_share(available: int, worker_count: int, expected: int) -> None:
    assert conftest.worker_share(available, worker_count) == expected


def test_no_cap_when_xdist_is_absent(
    monkeypatch: pytest.MonkeyPatch, _clear_pool_env: None
) -> None:
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    assert conftest.apply_worker_thread_cap() is None
    assert all(var not in os.environ for var in POOL_ENV_VARS)


def test_torch_pool_takes_the_share_under_xdist() -> None:
    worker_count = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if not worker_count:
        pytest.skip("not running under xdist")
    torch = pytest.importorskip("torch")
    share = conftest.worker_share(conftest.available_cpus(), int(worker_count))
    assert torch.get_num_threads() == share
