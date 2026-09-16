"""Co-located synthetic open-pit mine benchmark.

Generates a *single* ground-truth world in one coordinate system so that the
MWD, seismic and UCS sources are strictly co-located -- something no public
dataset provides.  The scene matches the report: X 0-500 m, Y 0-400 m,
elevation 0 to -120 m, 10 benches, a handful of drill holes.

Design principles
-----------------
* A latent "rock competence" field drives *both* UCS and acoustic impedance, so
  ``AI`` and ``UCS`` are correlated (calibration is meaningful) but not
  identical (an independent component keeps the seismic-derived strength
  imperfect -> fusion has something to gain).
* MWD parameters are produced from UCS through monotonic physical relations plus
  realistic scatter, so ``MWD -> UCS`` is learnable but noisy.
* Everything is derived from a fixed seed and hidden ground truth, enabling
  quantitative evaluation of every branch and of the fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

UCS_MIN, UCS_RANGE = 20.0, 150.0  # MPa
AI_MIN, AI_RANGE = 4.0e6, 7.0e6  # kg/(m^2 s)


@dataclass
class MineDataset:
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray  # elevations (negative, top=0)
    ucs_true: np.ndarray  # (nx, ny, nz) MPa
    ai_true: np.ndarray  # (nx, ny, nz)
    hole_ix: np.ndarray  # sample grid indices
    hole_iy: np.ndarray
    hole_iz: np.ndarray
    hole_xyz: np.ndarray  # (m, 3) physical coords of samples
    V: np.ndarray
    N: np.ndarray
    M: np.ndarray
    F: np.ndarray
    ucs_at_holes: np.ndarray  # true UCS at hole samples (core measurement)
    meta: dict = field(default_factory=dict)


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / (a.max() + 1e-12)


def generate_mine(
    shape: tuple[int, int, int] = (25, 20, 48),
    extent=((0.0, 500.0), (0.0, 400.0), (0.0, -120.0)),
    n_holes: int = 12,
    hole_sample_step: int = 1,
    seed: int = 42,
) -> MineDataset:
    nx, ny, nz = shape
    (x0, x1), (y0, y1), (z0, z1) = extent
    rng = np.random.default_rng(seed)

    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    gz = np.linspace(z0, z1, nz)  # 0 -> -120

    # --- latent competence field ------------------------------------------
    depth_frac = _normalize(-gz)[None, None, :] * np.ones(shape)  # 0 top -> 1 bottom
    bench = np.floor(depth_frac * 10.0)
    bench_offset = rng.uniform(-0.05, 0.05, size=11)[bench.astype(int)]

    smooth = _normalize(gaussian_filter(rng.standard_normal(shape), sigma=(3, 3, 4)))

    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    plane = 0.5 * xx / x1 + 0.5 * yy / y1 - (-zz) / 120.0
    fractured = np.exp(-((plane - 0.05) ** 2) / (2 * 0.03**2))

    competence = np.clip(
        0.15 + 0.55 * depth_frac + bench_offset + 0.20 * smooth - 0.35 * fractured,
        0.02, 1.0,
    )

    ucs_true = UCS_MIN + UCS_RANGE * competence

    independent = _normalize(gaussian_filter(rng.standard_normal(shape), sigma=(4, 4, 6)))
    ai_latent = np.clip(0.8 * competence + 0.2 * independent, 0.0, 1.0)
    ai_true = AI_MIN + AI_RANGE * ai_latent

    # --- drill holes -------------------------------------------------------
    hi = rng.choice(np.arange(1, nx - 1), size=n_holes, replace=False)
    hj = rng.choice(np.arange(1, ny - 1), size=n_holes, replace=False)
    depth_idx = np.arange(0, nz, hole_sample_step)

    ix, iy, iz = [], [], []
    for a, b in zip(hi, hj):
        for c in depth_idx:
            ix.append(a)
            iy.append(b)
            iz.append(c)
    ix = np.array(ix)
    iy = np.array(iy)
    iz = np.array(iz)

    ucs_holes = ucs_true[ix, iy, iz]
    ucs_norm = (ucs_holes - UCS_MIN) / UCS_RANGE

    # --- synthetic MWD from UCS (monotonic physics + scatter) --------------
    V = 2.0 * np.exp(-1.2 * ucs_norm) + 0.06 * rng.standard_normal(ucs_norm.shape)
    V = np.clip(V, 0.15, None)  # m/min
    N = 80.0 + 3.0 * rng.standard_normal(ucs_norm.shape)  # rpm
    M = 500.0 + 1500.0 * ucs_norm + 120.0 * rng.standard_normal(ucs_norm.shape)  # N.m
    F = 5000.0 + 20000.0 * ucs_norm + 1500.0 * rng.standard_normal(ucs_norm.shape)  # N

    hole_xyz = np.column_stack([gx[ix], gy[iy], gz[iz]])

    return MineDataset(
        gx=gx, gy=gy, gz=gz,
        ucs_true=ucs_true, ai_true=ai_true,
        hole_ix=ix, hole_iy=iy, hole_iz=iz, hole_xyz=hole_xyz,
        V=V, N=N, M=M, F=F, ucs_at_holes=ucs_holes,
        meta={"shape": shape, "extent": extent, "n_holes": n_holes, "seed": seed},
    )
