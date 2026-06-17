"""Plan-conditioned multi-modal world model on the token substrate.

A *world model* here is a learned action -> observation forward predictor.
The "action" is the **pulse schedule** — the pre-programmed actuator
waveforms an operator sets before a MAST shot (demanded plasma current,
density, coil/gas setpoints).  Given that plan plus a short initial-condition
window of tokenised diagnostics, the model autoregressively rolls every
diagnostic token stream forward in time.

The four pieces (each a module here):

* :mod:`~imas_ambix.worldmodel.dataset` — assemble per-shot multi-modal token
  sequences on a COMMON model time grid plus the pulse-schedule conditioning,
  split into a context window + a target window.  Every input path is routed
  through :func:`imas_ambix.tokenizer.store_targets.assert_not_target_path`
  so an eval-only reconstruction target can never enter the input stream.
* :mod:`~imas_ambix.worldmodel.model` — a plan-conditioned decoder-only
  Transformer with PER-GROUP-LOCAL token embeddings and the pulse schedule
  injected as PREPENDED conditioning tokens.
* :mod:`~imas_ambix.worldmodel.train` — a next-token NLL (cross-entropy)
  teacher-forcing training loop with the repo GPU-safety pattern, plus a
  tiny-overfit entrypoint that proves the wiring end-to-end.
* :mod:`~imas_ambix.worldmodel.eval` — autoregressive rollout of a held-out
  shot from its plan + initial window, scored predict-vs-reality against a
  persistence baseline (and structured against the eval-only L2 targets).

The substrate (token stores, registry, target boundary guard, L2 input
build) is read-only here — see the modules under
:mod:`imas_ambix.tokenizer` and :mod:`imas_ambix.data`.
"""

from __future__ import annotations

from imas_ambix.worldmodel.dataset import (
    ModalitySpec,
    WorldModelDataset,
    WorldModelSample,
    WorldModelWindowConfig,
    build_shot_sample,
    default_modalities,
    discover_worldmodel_shots,
)
from imas_ambix.worldmodel.model import (
    WorldModel,
    WorldModelConfig,
)

__all__ = [
    "ModalitySpec",
    "WorldModel",
    "WorldModelConfig",
    "WorldModelDataset",
    "WorldModelSample",
    "WorldModelWindowConfig",
    "build_shot_sample",
    "default_modalities",
    "discover_worldmodel_shots",
]
