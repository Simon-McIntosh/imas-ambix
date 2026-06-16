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
