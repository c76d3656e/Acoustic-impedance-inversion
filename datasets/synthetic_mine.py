"""Co-located synthetic open-pit mine benchmark.

Generates a *single* ground-truth world in one coordinate system so that the
MWD, seismic and UCS sources are strictly co-located -- something no public
dataset provides.  Default scene: a compact blast block X 0-50 m, Y 0-80 m,
elevation 0 to -40 m (about 10 benches).  Fourteen 梅花 holes pack into a
centred 20×50 m working face (three X-stations, with the middle column filled);
fusion maps crop to that window.

Design principles
-----------------
* A latent "rock competence" field drives *both* UCS and acoustic impedance, so
  ``AI`` and ``UCS`` are correlated (calibration is meaningful) but not
  identical.  UCS also carries a *mechanical residual* (weathering / alteration)
  that MWD and core see and AI does not; AI carries independent acoustic
  texture.  Without that split, ``corr(AI, UCS)`` collapses to ~0.98 and the
  borehole branch looks redundant on maps.
* MWD parameters are produced from UCS through monotonic physical relations plus
  realistic scatter, so ``MWD -> UCS`` is learnable but noisy.
* Drill holes follow an approximate triangular / 梅花 lattice packed into the
  default 20×50 m working face so the face is sampled uniformly rather than
  clustered, while the 50×80 m block remains available for panning.
* Everything is derived from a fixed seed and hidden ground truth, enabling
  quantitative evaluation of every branch and of the fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

UCS_MIN, UCS_RANGE = 20.0, 150.0  # MPa
AI_MIN, AI_RANGE = 4.0e6, 7.0e6  # kg/(m^2 s)

# Default working face inside the 50×80 m block. Twelve 梅花 holes pack here;
# fusion maps and the frontend crop to this window (the full block is still
# generated so the view can pan).
FACE_WIDTH = 20.0
FACE_HEIGHT = 50.0


def centered_face(
    gx: np.ndarray,
    gy: np.ndarray,
    width: float = FACE_WIDTH,
    height: float = FACE_HEIGHT,
) -> tuple[float, float, float, float]:
    """Return ``(x0, x1, y0, y1)`` of a window centred on the grid extent."""
    x0g, x1g = float(gx[0]), float(gx[-1])
    y0g, y1g = float(gy[0]), float(gy[-1])
    lx = min(float(width), max(x1g - x0g, 0.0))
    ly = min(float(height), max(y1g - y0g, 0.0))
    x0 = x0g + 0.5 * (x1g - x0g - lx)
    y0 = y0g + 0.5 * (y1g - y0g - ly)
    return (x0, x0 + lx, y0, y0 + ly)


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


def _farthest_point_order(
    pts: list[tuple[int, int]],
    gx: np.ndarray,
    gy: np.ndarray,
    n: int,
    start_xy: tuple[float, float],
) -> list[tuple[int, int]]:
    """Select ``n`` sites by farthest-point sampling, starting nearest ``start_xy``.

    Prefixes of the returned list stay spread (uniform nested sampling), which
    the 1→N borehole experiment needs.
    """
    if not pts or n < 1:
        return []
    n = min(int(n), len(pts))
    xy = np.array([(float(gx[i]), float(gy[j])) for i, j in pts], dtype=float)
    remaining = set(range(len(pts)))
    d0 = (xy[:, 0] - start_xy[0]) ** 2 + (xy[:, 1] - start_xy[1]) ** 2
    first = int(np.argmin(d0))
    order = [first]
    remaining.remove(first)
    while remaining and len(order) < n:
        chosen = xy[order]
        best_i = next(iter(remaining))
        best_d = -1.0
        for i in remaining:
            d = float(np.min(
                (chosen[:, 0] - xy[i, 0]) ** 2 + (chosen[:, 1] - xy[i, 1]) ** 2
            ))
            if d > best_d:
                best_d = d
                best_i = i
        order.append(best_i)
        remaining.remove(best_i)
    return [pts[i] for i in order]


def plum_blossom_hole_indices(
    gx: np.ndarray,
    gy: np.ndarray,
    n_holes: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[tuple[int, int]]:
    """Approximate triangular / 梅花 blast-hole lattice, snapped to the grid.

    Adjacent rows are shifted by half a hole spacing (equilateral-triangle
    packing, not a rectangular grid).  The lattice is packed into ``bbox``
    ``(x0, x1, y0, y1)`` when given, otherwise the full ``gx`` × ``gy``
    extent, then inset from that domain so wells sit *inside* it, not on the
    edges.  If snapping collapses a few sites, the remainder is filled by
    farthest-point sampling inside that inset.  The returned order is itself
    farthest-point from the domain centre, so every nested prefix is still spread.
    """
    if n_holes < 1:
        raise ValueError("n_holes must be >= 1")
    nx, ny = len(gx), len(gy)
    if nx < 3 or ny < 3:
        raise ValueError("grid too small for interior holes")
    if bbox is None:
        x_lo, x_hi = float(gx[0]), float(gx[-1])
        y_lo, y_hi = float(gy[0]), float(gy[-1])
    else:
        x_lo, x_hi, y_lo, y_hi = (float(v) for v in bbox)
        x_lo = max(x_lo, float(gx[0]))
        x_hi = min(x_hi, float(gx[-1]))
        y_lo = max(y_lo, float(gy[0]))
        y_hi = min(y_hi, float(gy[-1]))
    Lx = max(x_hi - x_lo, 1e-6)
    Ly = max(y_hi - y_lo, 1e-6)
    # Tall narrow face (20×50 m): keep the 3-station 梅花 spacing of 12 wells
    # and spend extra requested holes on the centre column (the two sites FPS
    # previously dropped) instead of tightening the whole lattice.
    n_pack = int(n_holes)
    if Ly >= 1.6 * Lx and n_holes >= 13:
        n_pack = 12
    dxg = abs(float(gx[1] - gx[0])) if nx > 1 else 1.0
    dyg = abs(float(gy[1] - gy[0])) if ny > 1 else 1.0
    a0 = float(np.sqrt((Lx * Ly / float(n_pack)) * 2.0 / np.sqrt(3.0)))
    # ~⅓ hole-spacing off each free face of the packing domain.  Cap at 12 %
    # so a 20×50 m working face still has room for 12 staggered wells.
    mx = min(max(0.30 * a0, 1.0 * dxg), 0.12 * Lx)
    my = min(max(0.30 * a0, 1.0 * dyg), 0.12 * Ly)
    x0 = x_lo + mx
    x1 = x_hi - mx
    y0 = y_lo + my
    y1 = y_hi - my
    if x1 <= x0 or y1 <= y0:
        raise ValueError("grid too small for interior holes")
    lx = max(x1 - x0, 1e-6)
    ly = max(y1 - y0, 1e-6)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    ix_lo = int(np.clip(np.argmin(np.abs(gx - x0)), 0, nx - 1))
    ix_hi = int(np.clip(np.argmin(np.abs(gx - x1)), 0, nx - 1))
    iy_lo = int(np.clip(np.argmin(np.abs(gy - y0)), 0, ny - 1))
    iy_hi = int(np.clip(np.argmin(np.abs(gy - y1)), 0, ny - 1))
    ix_lo = int(np.clip(ix_lo, 1, nx - 2))
    ix_hi = int(np.clip(ix_hi, 1, nx - 2))
    iy_lo = int(np.clip(iy_lo, 1, ny - 2))
    iy_hi = int(np.clip(iy_hi, 1, ny - 2))
    if ix_hi < ix_lo:
        ix_lo, ix_hi = ix_hi, ix_lo
    if iy_hi < iy_lo:
        iy_lo, iy_hi = iy_hi, iy_lo

    def _snap(x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(np.argmin(np.abs(gx - x)), ix_lo, ix_hi))
        iy = int(np.clip(np.argmin(np.abs(gy - y)), iy_lo, iy_hi))
        return ix, iy

    if n_holes == 1:
        return [_snap(cx, cy)]

    # Hexagonal packing inside the inset: area per hole ≈ a² √3 / 2.
    a = float(np.sqrt((lx * ly / float(n_pack)) * 2.0 / np.sqrt(3.0)))
    a = max(a, 1e-6)

    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(x: float, y: float) -> None:
        key = _snap(x, y)
        if key not in seen:
            seen.add(key)
            pts.append(key)

    min_stations = 3 if n_pack >= 6 else 2
    if ly >= lx:
        # Rows along X, holes along the long Y side; odd rows shifted by a/2.
        dx = a * np.sqrt(3.0) / 2.0
        n_row = max(min_stations, int(np.round(lx / max(dx, 1e-6))))
        n_col = max(min_stations, int(np.round(ly / max(a, 1e-6))))
        xs = np.linspace(x0, x1, n_row)
        even_ys = np.linspace(y0, y1, n_col)
        half = 0.5 * (even_ys[1] - even_ys[0]) if n_col > 1 else 0.0
        for r, x in enumerate(xs):
            if r % 2 == 0 or n_col == 1:
                ys = even_ys
            else:
                ys = even_ys[:-1] + half
            for y in np.atleast_1d(ys):
                _add(float(x), float(y))
    else:
        dy = a * np.sqrt(3.0) / 2.0
        n_col = max(min_stations, int(np.round(ly / max(dy, 1e-6))))
        n_row = max(min_stations, int(np.round(lx / max(a, 1e-6))))
        ys = np.linspace(y0, y1, n_col)
        even_xs = np.linspace(x0, x1, n_row)
        half = 0.5 * (even_xs[1] - even_xs[0]) if n_row > 1 else 0.0
        for c, y in enumerate(ys):
            if c % 2 == 0 or n_row == 1:
                xs = even_xs
            else:
                xs = even_xs[:-1] + half
            for x in np.atleast_1d(xs):
                _add(float(x), float(y))

    if len(pts) < n_holes:
        pool = [
            (i, j)
            for i in range(ix_lo, ix_hi + 1)
            for j in range(iy_lo, iy_hi + 1)
            if (i, j) not in seen
        ]
        while len(pts) < n_holes and pool:
            def _min_d2(p, placed=pts):
                return min(
                    (float(gx[p[0]]) - float(gx[q[0]])) ** 2
                    + (float(gy[p[1]]) - float(gy[q[1]])) ** 2
                    for q in placed
                )
            best = max(pool, key=_min_d2)
            pts.append(best)
            pool.remove(best)
            seen.add(best)

    return _farthest_point_order(pts, gx, gy, n=n_holes, start_xy=(cx, cy))


def generate_mine(
    shape: tuple[int, int, int] = (25, 40, 40),
    extent=((0.0, 50.0), (0.0, 80.0), (0.0, -40.0)),
    n_holes: int = 14,
    hole_sample_step: int = 1,
    seed: int = 42,
    hole_window: tuple[float, float] | None = (FACE_WIDTH, FACE_HEIGHT),
) -> MineDataset:
    nx, ny, nz = shape
    (x0, x1), (y0, y1), (z0, z1) = extent
    rng = np.random.default_rng(seed)
    z_span = abs(float(z1) - float(z0)) or 1.0

    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    gz = np.linspace(z0, z1, nz)

    # --- latent competence field ------------------------------------------
    # Strong, multi-scale LATERAL heterogeneity so horizontal (bench) slices
    # are not uniform; the depth trend is kept but no longer dominates.
    depth_frac = _normalize(-gz)[None, None, :] * np.ones(shape)  # 0 top -> 1 bottom
    bench = np.floor(depth_frac * 10.0)
    bench_offset = rng.uniform(-0.09, 0.09, size=11)[bench.astype(int)]

    # Fine scale is shorter than the ~18 m hole spacing so kriging between
    # 梅花 holes stays visibly smoother than the truth.
    fine = _normalize(gaussian_filter(rng.standard_normal(shape), sigma=(1.1, 1.1, 1.6)))
    med = _normalize(gaussian_filter(rng.standard_normal(shape), sigma=(3.0, 3.0, 4.0)))
    coarse = _normalize(gaussian_filter(rng.standard_normal(shape), sigma=(6.5, 6.5, 7.0)))

    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")

    def _fracture(ax, ay, az, c, w):
        plane = ax * xx / x1 + ay * yy / y1 + az * (-zz) / z_span
        return np.exp(-((plane - c) ** 2) / (2.0 * w**2))

    fr1 = _fracture(0.6, 0.5, -0.7, 0.15, 0.045)
    fr2 = _fracture(-0.4, 0.7, 0.5, 0.55, 0.05)
    fr3 = _fracture(0.2, -0.8, 0.3, 0.72, 0.04)

    def _blob(fx, fy, fz, rx, ry, rz):
        return np.exp(
            -(((xx - fx * x1) / rx) ** 2
              + ((yy - fy * y1) / ry) ** 2
              + ((zz + fz * z_span) / rz) ** 2)
        )

    hi1 = _blob(0.72, 0.38, 0.45, 0.14 * x1, 0.12 * y1, 0.14 * z_span)
    hi2 = _blob(0.22, 0.78, 0.72, 0.12 * x1, 0.10 * y1, 0.12 * z_span)
    lo1 = _blob(0.50, 0.58, 0.30, 0.15 * x1, 0.13 * y1, 0.12 * z_span)
    lo2 = _blob(0.82, 0.82, 0.55, 0.11 * x1, 0.10 * y1, 0.11 * z_span)

    competence = (
        0.06
        + 0.20 * depth_frac
        + bench_offset
        + 0.18 * med + 0.26 * fine + 0.07 * coarse
        + 0.55 * hi1 + 0.42 * hi2
        - 0.72 * fr1 - 0.62 * fr2 - 0.48 * fr3
        - 0.50 * lo1 - 0.38 * lo2
    )
    competence = np.clip((competence - 0.5) * 2.0 + 0.5, 0.02, 1.0)

    # Mechanical residual acoustics cannot see (weathering + alteration halo).
    # Keep this strong so S_Z / S_M / S_true disagree on maps, not only along holes.
    weather = (1.0 - depth_frac) ** 1.25
    alter = _blob(0.34, 0.30, 0.38, 0.22 * x1, 0.18 * y1, 0.22 * z_span)
    mech = np.clip(0.48 * weather + 1.05 * alter, 0.0, 1.0)
    ucs_latent = np.clip(0.52 * competence - 0.50 * mech, 0.02, 1.0)
    ucs_true = UCS_MIN + UCS_RANGE * ucs_latent

    ai_tex = _normalize(gaussian_filter(rng.standard_normal(shape), sigma=(1.8, 1.8, 2.6)))
    ai_latent = np.clip(0.40 * competence + 0.60 * ai_tex, 0.0, 1.0)
    ai_true = AI_MIN + AI_RANGE * ai_latent

    # --- drill holes: 梅花 / triangular lattice (uniform, not clustered) ---
    alter_xy = (0.34 * x1, 0.30 * y1)
    hard_xy = (0.72 * x1, 0.38 * y1)
    if hole_window is None:
        hole_bbox = (float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1]))
        lattice = plum_blossom_hole_indices(gx, gy, n_holes)
    else:
        hole_bbox = centered_face(gx, gy, float(hole_window[0]), float(hole_window[1]))
        lattice = plum_blossom_hole_indices(gx, gy, n_holes, bbox=hole_bbox)
    hi = np.array([p[0] for p in lattice], dtype=int)
    hj = np.array([p[1] for p in lattice], dtype=int)
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
        meta={
            "shape": shape, "extent": extent, "n_holes": n_holes, "seed": seed,
            "corr_ai_ucs": float(np.corrcoef(ai_true.ravel(), ucs_true.ravel())[0, 1]),
            "alter_xy": alter_xy,
            "hard_xy": hard_xy,
            "hole_layout": "plum_blossom",
            "hole_window": None if hole_window is None else (float(hole_window[0]), float(hole_window[1])),
            "hole_bbox": hole_bbox,
        },
    )


def unique_hole_xy_indices(ds: MineDataset) -> np.ndarray:
    """Unique hole ``(ix, iy)`` pairs in first-seen order, shape ``(n_holes, 2)``."""
    pairs = np.stack([np.asarray(ds.hole_ix), np.asarray(ds.hole_iy)], axis=1)
    _, first = np.unique(pairs, axis=0, return_index=True)
    return pairs[np.sort(first)]


def subset_holes(ds: MineDataset, n_holes: int) -> MineDataset:
    """Keep the first ``n_holes`` unique drill holes (nested prefix of the set).

    Sample arrays (MWD parameters, UCS, coordinates) are filtered to those
    holes; the 3-D ground-truth volumes are unchanged.  Prefix nesting means
    the ``k``-hole subset is contained in the ``k+1``-hole subset, which is
    what the well-count series experiment needs.
    """
    pairs = unique_hole_xy_indices(ds)
    n_total = len(pairs)
    if n_holes < 1 or n_holes > n_total:
        raise ValueError(f"n_holes must be in 1..{n_total}, got {n_holes}")
    keep = pairs[:n_holes]
    sample_pairs = np.stack([np.asarray(ds.hole_ix), np.asarray(ds.hole_iy)], axis=1)
    mask = (sample_pairs[:, None, :] == keep[None, :, :]).all(axis=2).any(axis=1)
    meta = dict(ds.meta)
    meta["n_holes"] = int(n_holes)
    return MineDataset(
        gx=ds.gx, gy=ds.gy, gz=ds.gz,
        ucs_true=ds.ucs_true, ai_true=ds.ai_true,
        hole_ix=ds.hole_ix[mask],
        hole_iy=ds.hole_iy[mask],
        hole_iz=ds.hole_iz[mask],
        hole_xyz=ds.hole_xyz[mask],
        V=ds.V[mask], N=ds.N[mask], M=ds.M[mask], F=ds.F[mask],
        ucs_at_holes=ds.ucs_at_holes[mask],
        meta=meta,
    )
