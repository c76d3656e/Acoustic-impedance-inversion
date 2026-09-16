"""Late fusion of the MWD and seismic strength fields.

Pipeline:

1. Calibrate impedance to strength ``UCS = f(AI)`` on co-located samples
   (:class:`ImpedanceStrengthCalibrator`), turning the seismic impedance volume
   into a second strength field ``S_Z`` with its own uncertainty.
2. Fuse ``S_MWD`` and ``S_Z`` with precision (inverse-variance) weighting so the
   locally more reliable source dominates automatically.
"""

from .calibration import ImpedanceStrengthCalibrator
from .uncertainty_fusion import (
    precision_weighted_fusion,
    simple_weighted_fusion,
)

__all__ = [
    "ImpedanceStrengthCalibrator",
    "precision_weighted_fusion",
    "simple_weighted_fusion",
]
