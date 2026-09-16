"""Impedance -> strength calibration ``UCS = f(AI)``.

Uses a Gaussian process so the mapping is non-parametric *and* yields a
predictive standard deviation that propagates into the fused uncertainty.  The
GP is trained only where both AI and UCS are available (well / core / co-located
calibration points).
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


class ImpedanceStrengthCalibrator:
    """Non-parametric ``UCS = f(AI)`` calibrator with uncertainty."""

    def __init__(self, random_state: int = 0):
        self._scaler = StandardScaler()
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e5))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e3))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e4))
        )
        self._gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
            random_state=random_state,
        )

    def fit(self, ai: np.ndarray, ucs: np.ndarray) -> "ImpedanceStrengthCalibrator":
        ai = np.asarray(ai, dtype=float).reshape(-1, 1)
        ucs = np.asarray(ucs, dtype=float).ravel()
        self._gpr.fit(self._scaler.fit_transform(ai), ucs)
        return self

    def predict(self, ai: np.ndarray, return_std: bool = True):
        ai = np.asarray(ai, dtype=float)
        flat = ai.reshape(-1, 1)
        Xs = self._scaler.transform(flat)
        if return_std:
            mu, std = self._gpr.predict(Xs, return_std=True)
            return mu.reshape(ai.shape), std.reshape(ai.shape)
        return self._gpr.predict(Xs).reshape(ai.shape)
