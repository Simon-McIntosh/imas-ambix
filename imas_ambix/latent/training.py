"""Corpus training: ONE shared hybrid-latent encoder across ALL campaigns.

The encoder is machine-agnostic by design — it maps generic absolute-calibrated
features to the hybrid latent regardless of which campaign's device geometry
produced them. Only the GS observation operator and the transport prior are
per-campaign (they are tied to that campaign's fixed sensor/coil geometry).
:class:`CorpusTrainer` trains the ONE shared encoder against every campaign's
:class:`~imas_ambix.latent.engine.GSGroundedLatentEngine` in the same
optimiser step, so gradients from every campaign's raw magnetics update the
same encoder weights — exactly what "machine-agnostic" requires.

A duplicate-parameter trap is easy to fall into here: naively concatenating
``engine.parameters()`` from N per-campaign engines that all wrap the *same*
encoder instance would register the encoder's parameters N times in the
optimiser, silently applying its update N times per step. :class:`CorpusTrainer`
builds its parameter set explicitly (the shared encoder once, each campaign's
transport once — :class:`~imas_ambix.latent.gs_observation.GSObservation` has
no learnable parameters, only fixed geometry buffers) to avoid this.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imas_ambix.latent.encoder import HybridLatentEncoder  # noqa: TC001
from imas_ambix.latent.engine import GSGroundedLatentEngine  # noqa: TC001


def consecutive_pairs(
    times: np.ndarray, *, max_dt: float | None = None
) -> list[tuple[int, int, float]]:
    """Build ``(t_index, t+1_index, dt)`` training pairs from ordered slice times.

    A pair is only formed between adjacent array indices whose time gap is
    ``<= max_dt`` (default: 1.5× the median step) — this prevents bridging a
    plasma-on discontinuity (a gap in the slice stream) into a spurious
    transport-prior transition.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size < 2:
        return []
    diffs = np.diff(t)
    if max_dt is None:
        max_dt = 1.5 * float(np.median(diffs))
    return [
        (i, i + 1, float(diffs[i])) for i in range(diffs.size) if diffs[i] <= max_dt
    ]


class CorpusTrainer:
    """Shared-encoder optimiser over a dict of per-campaign engines.

    ``engines`` : ``{campaign_key: GSGroundedLatentEngine}``, all wrapping the
    SAME ``encoder`` instance (passed in separately so the optimiser can be
    built correctly — see the module docstring for why this matters).
    """

    def __init__(
        self,
        encoder: HybridLatentEncoder,
        engines: dict[str, GSGroundedLatentEngine],
        *,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        self.encoder = encoder
        self.engines = dict(engines)
        params: list[torch.nn.Parameter] = list(encoder.parameters())
        seen = {id(p) for p in params}
        for engine in self.engines.values():
            for p in engine.transport.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.step_count = 0

    def to(self, device: torch.device | str) -> CorpusTrainer:
        self.encoder.to(device)
        for engine in self.engines.values():
            engine.gs.to(device)
            engine.transport.to(device)
        return self

    def step(
        self, batch_fns: dict[str, Callable[[], dict[str, Any]]]
    ) -> dict[str, float]:
        """One optimiser step: sum each campaign's composite loss, backward once.

        ``batch_fns`` : ``{campaign_key: () -> batch_dict}`` — called lazily so a
        campaign can be skipped (empty dict) without building its batch.
        """
        self.optimizer.zero_grad()
        totals: dict[str, float] = {}
        loss_sum: torch.Tensor | None = None
        for key, fn in batch_fns.items():
            batch = fn()
            if not batch:
                continue
            out = self.engines[key].losses(batch)
            totals[key] = float(out["total"].item())
            loss_sum = out["total"] if loss_sum is None else loss_sum + out["total"]
        if loss_sum is not None:
            loss_sum.backward()
            self.optimizer.step()
        self.step_count += 1
        return totals

    def save(
        self,
        path: str | Path,
        *,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Checkpoint the encoder + per-campaign transport + optimiser state.

        ``extra`` carries arbitrary picklable metadata (e.g. the corpus-level
        feature / anchored / command normalisation stats) that must survive
        alongside the weights — a checkpoint without its exact input scaling
        cannot be evaluated faithfully.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.step_count if step is None else step,
            "encoder": self.encoder.state_dict(),
            "transport": {k: e.transport.state_dict() for k, e in self.engines.items()},
            "optimizer": self.optimizer.state_dict(),
            "extra": extra or {},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)  # atomic on the same filesystem — SIGTERM-safe

    def load(
        self,
        path: str | Path,
        *,
        map_location: str | None = None,
        return_extra: bool = False,
    ) -> int | tuple[int, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        self.encoder.load_state_dict(payload["encoder"])
        for k, sd in payload["transport"].items():
            if k in self.engines:
                self.engines[k].transport.load_state_dict(sd)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step_count = int(payload["step"])
        if return_extra:
            return self.step_count, payload.get("extra", {})
        return self.step_count


__all__ = ["consecutive_pairs", "CorpusTrainer"]
