"""GPU round-trip test for OpenMagvit2Tokenizer.

Skipped automatically when CUDA is not available (login node) or when the
Open-MAGVIT2 staging directory is absent.  Run on the betelgeuse GPU node
via a SLURM allocation (--reservation=gpu_0003_grpA) to exercise the real
encode + decode path.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from pathlib import Path  # noqa: E402

from imas_ambix.tokenizer.frames import OpenMagvit2Tokenizer  # noqa: E402

MAGVIT_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/open-magvit2")
pytestmark = pytest.mark.skipif(
    not MAGVIT_ROOT.is_dir(), reason="Open-MAGVIT2 staging missing"
)


def test_cuda_round_trip_one_frame():
    """Encode + decode a synthetic 256x256 RGB frame on CUDA, assert shapes."""
    import numpy as np

    tok = OpenMagvit2Tokenizer(device="cuda")
    frames = np.random.randint(0, 256, size=(1, 256, 256, 3), dtype=np.uint8)
    enc = tok.encode(frames)
    dec = tok.decode(enc)
    assert enc.token_ids.shape == (1, 16, 16)
    assert dec.shape == (1, 256, 256, 3)
