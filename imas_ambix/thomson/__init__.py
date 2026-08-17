"""Thomson pseudo-magnetic observation models.

The pedestal calibration deliberately consumes chord-level ``psi_N`` sampled
from DIII-D train labels.  That is derivative-level label consumption under the
locked firewall: label maps never supervise an equilibrium map or a free-form
residual.  The isotherm-asymmetry operator is analytic and accepts no fitted
parameters or training data.
"""

from .bank import (
    BankedEquilibriumMoment,
    banked_equilibrium_moments,
    collect_pedestal_samples,
)
from .calibration import PedestalCalibration, PedestalFootDetector
from .models import (
    ChannelAssessment,
    ChannelValidityPolicy,
    ElmPhase,
    IsofluxPair,
    IsofluxPairer,
    IsothermAsymmetryOperator,
    SeparatrixEstimate,
    TopologyClass,
    ValidityReason,
)

__all__ = [
    "BankedEquilibriumMoment",
    "ChannelAssessment",
    "ChannelValidityPolicy",
    "ElmPhase",
    "IsofluxPair",
    "IsofluxPairer",
    "IsothermAsymmetryOperator",
    "PedestalCalibration",
    "PedestalFootDetector",
    "SeparatrixEstimate",
    "TopologyClass",
    "ValidityReason",
    "banked_equilibrium_moments",
    "collect_pedestal_samples",
]
