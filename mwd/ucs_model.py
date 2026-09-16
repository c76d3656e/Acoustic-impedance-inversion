"""Physics-Guided Gaussian Process Regression (PG-GPR) for MWD -> UCS.

A purely data-driven GPR struggles to extrapolate.  We inject physics by using
a linear specific-energy trend as the GP *mean function* and letting the GP
model only the residual.  Predictions therefore fall back to a sensible
physical baseline where data is sparse, and the GP still supplies a calibrated
predictive standard deviation used later for uncertainty-aware fusion.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class PhysicsGuidedGPR:
    """UCS regressor with a specific-energy physical mean + GP residual.

    Parameters
    ----------
    se_index:
        Column index of specific energy in the feature matrix (default 4,
        matching :data:`mwd.features.FEATURE_NAMES`).
    """

    def __init__(self, se_index: int = 4, random_state: int = 0):
        self.se_index = se_index
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._trend = LinearRegression()
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e5))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e3))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e4))
        )
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhysicsGuidedGPR":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        # Physical baseline: UCS ~ linear in specific energy.
        se = X[:, [self.se_index]]
        self._trend.fit(se, y)
        residual = y - self._trend.predict(se)
        Xs = self._scaler.fit_transform(X)
        self._gpr.fit(Xs, residual)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        """Return ``mu`` (and ``sigma`` when ``return_std``)."""
        X = np.asarray(X, dtype=float)
        se = X[:, [self.se_index]]
        base = self._trend.predict(se)
        Xs = self._scaler.transform(X)
        if return_std:
            res_mu, res_std = self._gpr.predict(Xs, return_std=True)
            return base + res_mu, res_std
        return base + self._gpr.predict(Xs)
