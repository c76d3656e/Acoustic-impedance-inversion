"""Synthetic-mine benchmark loader / cache.

Generates the co-located benchmark on demand (no download) and optionally caches
it to ``.npz`` so experiments reuse the exact same ground truth.
"""

from __future__ import annotations

import os

import numpy as np

from datasets.synthetic_mine import generate_mine, MineDataset


def load_synthetic_mine(
    cache: str = "data/synthetic_mine/mine.npz", regenerate: bool = False, **kwargs
) -> MineDataset:
    """Load (or generate and cache) the synthetic-mine benchmark."""
    if os.path.exists(cache) and not regenerate:
        d = np.load(cache, allow_pickle=True)
        return MineDataset(**{k: d[k] for k in d.files if k != "meta"},
                           meta=d["meta"].item())

    ds = generate_mine(**kwargs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(
        cache,
        gx=ds.gx, gy=ds.gy, gz=ds.gz, ucs_true=ds.ucs_true, ai_true=ds.ai_true,
        hole_ix=ds.hole_ix, hole_iy=ds.hole_iy, hole_iz=ds.hole_iz,
        hole_xyz=ds.hole_xyz, V=ds.V, N=ds.N, M=ds.M, F=ds.F,
        ucs_at_holes=ds.ucs_at_holes, meta=ds.meta,
    )
    return ds
