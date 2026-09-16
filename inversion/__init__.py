"""Acoustic impedance inversion algorithms.

This package implements the mathematical chain

    acoustic impedance  ->  reflectivity  ->  synthetic seismic  ->  inversion

used to recover acoustic impedance ``AI = rho * Vp`` from post-stack seismic
data.  All operators are built in the *log-impedance* domain, where reflectivity
becomes a linear difference operator and inversion reduces to a well-posed,
Tikhonov-regularised least-squares problem once a low-frequency background model
is supplied.
"""

from .wavelet import ricker
from .reflectivity import (
    reflectivity_from_impedance,
    impedance_from_reflectivity,
)
from .forward import (
    difference_operator,
    wavelet_matrix,
    forward_operator,
    synthetic_seismic,
    synthetic_seismic_volume,
)
from .model_based import invert_trace, invert_volume, background_model
from .sparse_spike import sparse_spike_inversion

__all__ = [
    "ricker",
    "reflectivity_from_impedance",
    "impedance_from_reflectivity",
    "difference_operator",
    "wavelet_matrix",
    "forward_operator",
    "synthetic_seismic",
    "synthetic_seismic_volume",
    "invert_trace",
    "invert_volume",
    "background_model",
    "sparse_spike_inversion",
]
