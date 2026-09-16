"""Data-warehouse loaders for the five project data modules.

| Module            | Loader                     | Source / status                         |
| ----------------- | -------------------------- | --------------------------------------- |
| MWD-UCS           | :func:`load_mwd_ucs`       | Sci. Reports 2025 (on request)          |
| MWD-spatial       | :func:`load_mwd_spatial`   | Hansen / Zenodo 10358374 (download)     |
| Marmousi2         | :func:`load_marmousi2`     | SEG / GitHub (download)                 |
| Penobscot         | :func:`load_penobscot`     | Zenodo 1325077 / SEG (download)         |
| synthetic-mine    | :func:`load_synthetic_mine`| generated locally (no download)         |

Real datasets are fetched on demand with ``scripts/download_datasets.py``; the
loaders raise a clear ``FileNotFoundError`` with the source URL when a file is
missing, so the offline pipeline never depends on external hosts.
"""

from .mwd_ucs import load_mwd_ucs
from .mwd_spatial import load_mwd_spatial
from .marmousi2 import load_marmousi2
from .penobscot import load_penobscot
from .synthetic_mine import load_synthetic_mine

__all__ = [
    "load_mwd_ucs",
    "load_mwd_spatial",
    "load_marmousi2",
    "load_penobscot",
    "load_synthetic_mine",
]
