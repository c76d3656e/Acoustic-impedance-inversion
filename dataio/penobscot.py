"""Penobscot 3D loader (Zenodo 1325077 / SEG / dGB).

Thin wrapper around :mod:`preprocessing` to read the post-stack SEG-Y cube and
the B-41 / L-30 well logs.  Returns raw arrays so the same inversion engine used
for the synthetic model can be applied to the field data.
"""

from __future__ import annotations

import os

SOURCE = "https://zenodo.org/records/1325077"


def load_penobscot(
    seismic_path: str = "data/penobscot/seismic/penobscot.sgy",
    well_paths=("data/penobscot/wells/B-41.las", "data/penobscot/wells/L-30.las"),
):
    """Return ``(cube, dt, wells)`` where ``wells`` maps path -> LASFile."""
    from preprocessing import read_segy, read_las

    if not os.path.exists(seismic_path):
        raise FileNotFoundError(
            f"Penobscot SEG-Y not found at {seismic_path}. Download from "
            f"{SOURCE} (see scripts/download_datasets.py --dataset penobscot)."
        )
    cube, dt = read_segy(seismic_path)
    wells = {p: read_las(p) for p in well_paths if os.path.exists(p)}
    return cube, dt, wells
