"""Error metrics comparing inverted and true acoustic impedance."""

from __future__ import annotations

import numpy as np


def _flat(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    return a, b


def r2_score(true, pred) -> float:
    """Coefficient of determination ``R^2``."""
    t, p = _flat(true, pred)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def rmse(true, pred) -> float:
    t, p = _flat(true, pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mae(true, pred) -> float:
    t, p = _flat(true, pred)
    return float(np.mean(np.abs(t - p)))


def error_volume(true, pred) -> np.ndarray:
    """Absolute error, preserving the input shape."""
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError("inputs must have the same shape")
    return np.abs(t - p)


def summary(true, pred) -> dict:
    """Convenience bundle of all scalar metrics."""
    return {"R2": r2_score(true, pred), "RMSE": rmse(true, pred), "MAE": mae(true, pred)}
