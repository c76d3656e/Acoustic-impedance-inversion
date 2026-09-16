"""MWD (Measurement-While-Drilling) branch.

Turns raw drilling parameters into engineered features (including Teale's
specific energy) and predicts rock strength (UCS) with a physics-guided
Gaussian-process regressor that also reports a predictive standard deviation --
essential for the downstream uncertainty-aware fusion.
"""

from .features import mwd_features, FEATURE_NAMES, specific_energy
from .ucs_model import PhysicsGuidedGPR

__all__ = ["mwd_features", "FEATURE_NAMES", "specific_energy", "PhysicsGuidedGPR"]
