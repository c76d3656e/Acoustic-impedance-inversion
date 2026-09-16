"""MWD spatial dataset loader (Hansen et al., Zenodo 10358374).

Measure-While-Drilling records with rock-type labels for 15 Norwegian hard-rock
tunnels.  Useful for realistic MWD parameter distributions and spatial modelling
(it has no UCS, so use it for MWD structure -- not as MWD->UCS ground truth).
"""

from __future__ import annotations

import os

import pandas as pd

SOURCE = "https://zenodo.org/records/10358374"


def load_mwd_spatial(path: str = "data/mwd_spatial/mwd_full.csv") -> pd.DataFrame:
    """Load a Hansen MWD CSV (raw/train/test/full) as a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MWD spatial CSV not found at {path}. Download from {SOURCE} "
            f"(see scripts/download_datasets.py --dataset mwd_spatial)."
        )
    return pd.read_csv(path)
