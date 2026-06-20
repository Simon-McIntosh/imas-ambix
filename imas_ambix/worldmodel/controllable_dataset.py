"""Camera windows + actuator-PLAN drive surface + optional measured observations.

Extends :mod:`imas_ambix.worldmodel.spacetime_dataset_v2` (camera frames + the
tokenised pulse-schedule plan + the measured diagnostic streams) with the
DEMANDED actuator PLAN as the always-on drive surface — the
``control-surface = actuator-vector`` decision.  The camera-frame target, the
tokenised plan prefix, the measured-signal conditioning, the windowing, and the
local-id rebasing are all reused from v2; this module only adds the actuator-plan
read (:func:`imas_ambix.worldmodel.actuator_plan.read_window_actuator_plan`) and
a sample / collate that carry the actuator drive alongside the v2 conditioning.

A controllable sample is a v2 :class:`SignalSpacetimeSample` plus the actuator
plan for the same window.  The actuator plan is the surface the operator edits to
"play" the plasma; the measured signals are OPTIONAL context (high-dropout at
training so the model cannot shortcut the control->camera map through the
redundant realised observations).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from imas_ambix.worldmodel.actuator_plan import (
    ACTUATOR_CHANNELS,
    ActuatorPlan,
    read_window_actuator_plan,
)
from imas_ambix.worldmodel.spacetime_dataset import REFERENCE_CAMERA
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    SignalSpacetimeSample,
    assemble_signal_window,
    default_signal_modalities,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np

    from imas_ambix.camdyn.conditioning import ConditioningChannel
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

logger = logging.getLogger(__name__)


@dataclass
class ControllableSpacetimeSample:
    """A v2 :class:`SignalSpacetimeSample` + the demanded actuator plan.

    Attributes
    ----------
    signal:
        The v2 camera window + tokenised plan + measured-signal conditioning.
    actuator:
        The demanded actuator plan (drive surface) for the SAME window.
    """

    signal: SignalSpacetimeSample
    actuator: ActuatorPlan

    # convenient pass-throughs so a controllable sample is a drop-in for the
    # v2 consumers (the gate reuses the v2 rollout helpers).
    @property
    def shot_id(self) -> int:
        return self.signal.shot_id

    @property
    def camera(self) -> str:
        return self.signal.camera

    @property
    def start_frame(self) -> int:
        return self.signal.start_frame

    @property
    def frames(self) -> np.ndarray:
        return self.signal.frames

    @property
    def plan(self) -> np.ndarray:
        return self.signal.plan

    @property
    def signals(self) -> dict[str, np.ndarray]:
        return self.signal.signals

    @property
    def frame_time(self) -> np.ndarray:
        return self.signal.frame_time

    @property
    def context_frames(self) -> int:
        return self.signal.context_frames

    @property
    def n_frames(self) -> int:
        return self.signal.n_frames


def assemble_controllable_window(
    shot_id: int,
    config: SpacetimeWindowConfig,
    modalities: Sequence[SignalModalitySpec],
    n_signal_steps: int,
    n_act_steps: int,
    *,
    camera: str = REFERENCE_CAMERA,
    token_root: Path | None = None,
    start_frame: int | None = None,
    actuator_channels: Sequence[ConditioningChannel] = ACTUATOR_CHANNELS,
) -> ControllableSpacetimeSample:
    """Assemble a v2 camera window + the demanded actuator plan for that window."""
    signal = assemble_signal_window(
        int(shot_id),
        config,
        modalities,
        n_signal_steps,
        camera=camera,
        token_root=token_root,
        start_frame=start_frame,
    )
    actuator = read_window_actuator_plan(
        int(shot_id),
        signal.base,
        int(n_act_steps),
        channels=actuator_channels,
    )
    return ControllableSpacetimeSample(signal=signal, actuator=actuator)


class ControllableSpacetimeDataset:
    """Map-style dataset of camera windows + actuator plan + measured signals.

    Mirrors :class:`imas_ambix.worldmodel.spacetime_dataset_v2.SignalSpacetimeDataset`
    (lazy Zarr-on-demand, worker-safe, optional random window jitter) and adds
    the per-window actuator-plan read.
    """

    def __init__(
        self,
        shot_ids: Sequence[int],
        config: SpacetimeWindowConfig,
        modalities: Sequence[SignalModalitySpec],
        n_signal_steps: int,
        n_act_steps: int,
        *,
        camera: str = REFERENCE_CAMERA,
        token_root: Path | None = None,
        random_window: bool = False,
        seed: int = 0,
        actuator_channels: Sequence[ConditioningChannel] = ACTUATOR_CHANNELS,
    ) -> None:
        self._shot_ids = [int(s) for s in shot_ids]
        self._config = config
        self._modalities = list(modalities)
        self._n_signal_steps = int(n_signal_steps)
        self._n_act_steps = int(n_act_steps)
        self._camera = camera
        self._token_root = token_root
        self._random = bool(random_window)
        self._seed = int(seed)
        self._actuator_channels = tuple(actuator_channels)

    def __len__(self) -> int:
        return len(self._shot_ids)

    def __getitem__(self, index: int) -> ControllableSpacetimeSample:
        from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
            camera_frame_count,
        )

        sid = self._shot_ids[index]
        start = None
        if self._random:
            import random  # noqa: PLC0415

            span = (self._config.n_frames - 1) * self._config.frame_stride + 1
            try:
                n_total = camera_frame_count(
                    sid, self._camera, token_root=self._token_root
                )
            except (FileNotFoundError, KeyError, ValueError):
                n_total = span
            hi = max(0, n_total - span)
            rng = random.Random((self._seed * 1_000_003) ^ (sid * 31) ^ index)
            start = rng.randint(0, hi) if hi > 0 else 0
        return assemble_controllable_window(
            sid,
            self._config,
            self._modalities,
            self._n_signal_steps,
            self._n_act_steps,
            camera=self._camera,
            token_root=self._token_root,
            start_frame=start,
            actuator_channels=self._actuator_channels,
        )

    @property
    def config(self) -> SpacetimeWindowConfig:
        return self._config


__all__ = [
    "ControllableSpacetimeDataset",
    "ControllableSpacetimeSample",
    "assemble_controllable_window",
    "default_signal_modalities",
]
