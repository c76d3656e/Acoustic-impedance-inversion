"""Feature engineering for MWD drilling parameters.

Raw parameters
--------------
* ``V`` : penetration / drilling rate (m/s or m/min)
* ``N`` : rotation speed (rev/s or rpm)
* ``M`` : torque (N.m)
* ``F`` : thrust / feed force (N)

Derived
-------
Teale-style specific energy (per the report's convention):

    SE' = F + 2*pi*N*M / V

plus dimensionless ratios that help separate lithology-driven responses from
operational drilling changes.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["V", "N", "M", "F", "SE_prime", "M_over_F", "V_over_N", "V_over_F"]


def specific_energy(V, N, M, F, eps: float = 1e-9) -> np.ndarray:
    """Specific energy ``SE' = F + 2*pi*N*M / V`` (element-wise)."""
    V = np.asarray(V, dtype=float)
    N = np.asarray(N, dtype=float)
    M = np.asarray(M, dtype=float)
    F = np.asarray(F, dtype=float)
    return F + 2.0 * np.pi * N * M / (V + eps)


def mwd_features(V, N, M, F, eps: float = 1e-9) -> np.ndarray:
    """Return the ``(n, 8)`` feature matrix in :data:`FEATURE_NAMES` order."""
    V = np.asarray(V, dtype=float)
    N = np.asarray(N, dtype=float)
    M = np.asarray(M, dtype=float)
    F = np.asarray(F, dtype=float)
    se = specific_energy(V, N, M, F, eps=eps)
    return np.column_stack(
        [
            V,
            N,
            M,
            F,
            se,
            M / (F + eps),
            V / (N + eps),
            V / (F + eps),
        ]
    )
