"""A self-contained, reproducible synthetic 3-D earth model.

Builds layered P-wave velocity and density volumes with gently folded,
laterally varying layer boundaries plus a localised high-impedance lens.  This
gives a non-trivial ground-truth acoustic-impedance volume whose horizontal
slices actually vary spatially -- ideal for validating the inversion end to end
without downloading gigabytes of field data.
"""

from __future__ import annotations

import numpy as np


def acoustic_impedance(vp: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Acoustic impedance ``AI = rho * Vp`` (units ``kg / (m^2 s)``)."""
    return np.asarray(rho, dtype=float) * np.asarray(vp, dtype=float)


def layered_property_model(
    shape: tuple[int, int, int] = (64, 64, 220),
    seed: int = 7,
):
    """Return ``(vp, rho)`` volumes of shape ``(nx, ny, nz)``.

    Velocities increase with depth across six folded layers; density follows
    Gardner's relation ``rho = 310 * Vp**0.25`` (SI).  A smooth high-impedance
    lens is embedded near the centre so horizontal slices show a clear anomaly.
    """
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    depth_idx = np.arange(nz)

    fracs = np.array([0.12, 0.28, 0.42, 0.58, 0.72, 0.86])
    amps = np.array([6.0, 8.0, 7.0, 10.0, 6.0, 5.0])
    vps = np.array([1800.0, 2200.0, 2600.0, 3100.0, 3500.0, 4000.0, 4400.0])

    vp = np.full(shape, vps[0], dtype=float)
    for k, (frac, amp) in enumerate(zip(fracs, amps)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        surf = (
            frac * nz
            + amp * np.sin(2.0 * np.pi * xx / nx * (1.0 + 0.3 * k) + phase)
            + 0.6 * amp * np.cos(2.0 * np.pi * yy / ny)
        )
        mask = depth_idx[None, None, :] >= surf[:, :, None]
        vp = np.where(mask, vps[k + 1], vp)

    vp *= 1.0 + 0.01 * rng.standard_normal(shape)

    cx, cy, cz = nx * 0.6, ny * 0.4, nz * 0.5
    gxy = np.exp(
        -(
            (xx - cx) ** 2 / (2.0 * (nx * 0.12) ** 2)
            + (yy - cy) ** 2 / (2.0 * (ny * 0.12) ** 2)
        )
    )
    gz = np.exp(-((depth_idx - cz) ** 2) / (2.0 * (nz * 0.04) ** 2))
    vp *= 1.0 + 0.18 * gxy[:, :, None] * gz[None, None, :]

    rho = 310.0 * vp**0.25
    return vp, rho
