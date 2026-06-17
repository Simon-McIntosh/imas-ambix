"""Shared fixtures for the HF-signal tokenizer tests.

The signal tokenizers and the encode driver allocate id ranges against the
shared :data:`imas_ambix.tokenizer.registry.registry` singleton.  Different
tests construct tokenizers with different vocab sizes for the same block
name, so without a reset the second allocation would clash.  Reset the
singleton in place before every test (swapping the module attribute would
not reach call sites that captured ``registry`` at import time).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_registry():
    from imas_ambix.tokenizer.registry import CONTROL_RANGE
    from imas_ambix.tokenizer.registry import registry as singleton

    singleton._blocks.clear()
    singleton._cursor = CONTROL_RANGE[1]
    yield
    singleton._blocks.clear()
    singleton._cursor = CONTROL_RANGE[1]


@pytest.fixture(autouse=True)
def _permissive_corpus_guards(monkeypatch):
    """Default the on-disk group-presence guard to permissive in unit tests.

    ``encode_shots`` consults ``group_present`` (a cheap on-disk group probe)
    before decoding.  Tests that monkeypatch ``load_shot_window`` with synthetic
    data have no real shot on disk, so default it to present; a test exercising
    the guard itself overrides it explicitly.  ``already_encoded`` is left as the
    real implementation — the encode tests redirect ``TOKEN_ROOT`` to a fresh
    tmp dir, so it correctly returns False for their synthetic shots.
    """
    from imas_ambix.tokenizer import signal_hf_encode as enc

    monkeypatch.setattr(enc, "group_present", lambda s, g: True)
    yield
