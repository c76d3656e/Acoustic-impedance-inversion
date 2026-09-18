"""Late fusion of the MWD and seismic strength fields.

Pipeline:

1. Calibrate impedance to strength ``UCS = f(AI)`` on co-located samples
   (:class:`ImpedanceStrengthCalibrator`), turning the seismic impedance volume
   into a second strength field ``S_Z`` with its own uncertainty.
2. Fuse the *interpolated* ``S_MWD`` with ``S_Z`` by precision weighting, then
   re-inject borehole hard data near the traces so impedance cannot cancel them.
"""

from .calibration import ImpedanceStrengthCalibrator
from .uncertainty_fusion import (
    precision_weighted_fusion,
    simple_weighted_fusion,
    borehole_anchor_weight,
    anchor_borehole_hard_data,
)
from .pipeline import run_fusion_pipeline, FusionResult

__all__ = [
    "ImpedanceStrengthCalibrator",
    "precision_weighted_fusion",
    "simple_weighted_fusion",
    "borehole_anchor_weight",
    "anchor_borehole_hard_data",
    "run_fusion_pipeline",
    "FusionResult",
]
