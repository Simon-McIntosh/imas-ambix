"""Calibration library for IMAS Ambix.

Provides corpus-wide statistics for signal channels and camera frames,
enabling principled normalisation in tokenizers and the training loop.

Public API::

    from imas_ambix.calibration import (
        ChannelCalibration,
        FrameCalibration,
        compute_signal_calibration,
        compute_frame_calibration,
        save_calibration,
        load_signal_calibration,
        load_frame_calibration,
        CALIBRATION_ROOT,
    )
"""

from __future__ import annotations

from imas_ambix.calibration.frames import FrameCalibration, compute_frame_calibration
from imas_ambix.calibration.persistence import (
    CALIBRATION_ROOT,
    load_frame_calibration,
    load_signal_calibration,
    save_calibration,
)
from imas_ambix.calibration.signals import (
    ChannelCalibration,
    compute_signal_calibration,
)

__all__ = [
    "ChannelCalibration",
    "FrameCalibration",
    "compute_signal_calibration",
    "compute_frame_calibration",
    "save_calibration",
    "load_signal_calibration",
    "load_frame_calibration",
    "CALIBRATION_ROOT",
]
