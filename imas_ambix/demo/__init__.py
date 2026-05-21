"""Demo pipeline for the WHAM Fusion World Model (plans/demo.md).

Provides :func:`run_demo` and :class:`DemoArtifacts` for running a
forward-prediction rollout on a held-out MAST shot, decoding predicted
frames, computing evaluation metrics, and writing the comparison artefacts
described in ``plans/demo.md`` §5.

Example usage::

    from pathlib import Path
    from imas_ambix.demo import run_demo, DemoArtifacts

    artefacts = run_demo(
        shot_id=30420,
        checkpoint_path="mock",
        output_dir=Path("/tmp/demo-30420"),
    )
    print(artefacts.metrics_json)
"""

from __future__ import annotations

from imas_ambix.demo.runner import DemoArtifacts, run_demo

__all__ = [
    "DemoArtifacts",
    "run_demo",
]
