"""Uncertainty-aware fusion of two strength fields.

Treating each source as a Gaussian estimate ``N(mu, sigma^2)``, the
precision-weighted (inverse-variance) combination is the maximum-likelihood
fusion for independent estimates:

    mu_f    = (mu_M/sigma_M^2 + mu_Z/sigma_Z^2) / (1/sigma_M^2 + 1/sigma_Z^2)
    sigma_f^2 = 1 / (1/sigma_M^2 + 1/sigma_Z^2)

The locally more reliable source (smaller sigma) automatically dominates.
"""

from __future__ import annotations

import numpy as np


def precision_weighted_fusion(mu_M, var_M, mu_Z, var_Z, eps: float = 1e-12):
    """Inverse-variance fusion. Returns ``(mu_f, var_f)`` (same shape)."""
    mu_M = np.asarray(mu_M, dtype=float)
    var_M = np.asarray(var_M, dtype=float)
    mu_Z = np.asarray(mu_Z, dtype=float)
    var_Z = np.asarray(var_Z, dtype=float)

    tau_M = 1.0 / (var_M + eps)
    tau_Z = 1.0 / (var_Z + eps)
    var_f = 1.0 / (tau_M + tau_Z)
    mu_f = (mu_M * tau_M + mu_Z * tau_Z) * var_f
    return mu_f, var_f


def simple_weighted_fusion(mu_M, mu_Z, w: float = 0.5):
    """Fixed-weight baseline: ``w * mu_M + (1 - w) * mu_Z``."""
    if not 0.0 <= w <= 1.0:
        raise ValueError("w must be in [0, 1]")
    return w * np.asarray(mu_M, dtype=float) + (1.0 - w) * np.asarray(mu_Z, dtype=float)
