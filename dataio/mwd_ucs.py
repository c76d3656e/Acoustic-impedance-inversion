"""MWD-UCS calibration data loader.

Expected CSV columns (case-insensitive): ``F, M, V, N, UCS`` -- the 197-sample
MWD/UCS table from Zhang et al., *Scientific Reports* 2025
(https://www.nature.com/articles/s41598-025-93111-4).  That data is available
on reasonable request to the authors, so it is not bundled here.
"""

from __future__ import annotations

import os

import pandas as pd

SOURCE = "https://www.nature.com/articles/s41598-025-93111-4"


def load_mwd_ucs(path: str = "data/mwd_ucs/mwd_ucs.csv") -> pd.DataFrame:
    """Load the MWD-UCS calibration table as a DataFrame with F/M/V/N/UCS."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MWD-UCS table not found at {path}. Request it from the authors of "
            f"{SOURCE} and save it there (columns: F, M, V, N, UCS)."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    required = {"F", "M", "V", "N", "UCS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MWD-UCS file missing columns: {sorted(missing)}")
    return df
