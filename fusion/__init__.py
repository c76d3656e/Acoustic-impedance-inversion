"""Late fusion of the MWD and seismic strength fields.

Pipeline:

1. Calibrate impedance to strength ``UCS = f(AI)`` on co-located samples
   (:class:`ImpedanceStrengthCalibrator`), turning the seismic impedance volume
   into a second strength field ``S_Z`` with its own uncertainty.
2. Fuse borehole kriging ``S_M`` with collocated ``S_Z`` by Doyen's Bayesian
   collocated cokriging (SPE 36498).  ``ρ`` is ``corr(UCS, AI)`` at the holes
   so a one-well GP cannot claim perfect correlation.  Hole voxels are
   written back to the MWD point estimates.
"""

from .calibration import ImpedanceStrengthCalibrator
from .uncertainty_fusion import (
    precision_weighted_fusion,
    simple_weighted_fusion,
    doyen_collocated_update,
    borehole_anchor_weight,
    anchor_borehole_hard_data,
)
from .pipeline import run_fusion_pipeline, FusionResult

__all__ = [
    "ImpedanceStrengthCalibrator",
    "precision_weighted_fusion",
    "simple_weighted_fusion",
    "doyen_collocated_update",
    "borehole_anchor_weight",
    "anchor_borehole_hard_data",
    "run_fusion_pipeline",
    "FusionResult",
]
