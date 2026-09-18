"""Synthetic and field-data model builders.

``synthetic`` provides a self-contained, reproducible 3-D earth model used for
Stage I validation (no external downloads required).  Real datasets
(Marmousi2, Penobscot 3D) can be fetched with ``scripts/download_datasets.py``
and loaded through the :mod:`preprocessing` package.
"""

from .synthetic import (
    layered_property_model,
    acoustic_impedance,
)
from .synthetic_mine import (
    generate_mine,
    MineDataset,
    unique_hole_xy_indices,
    subset_holes,
)

__all__ = [
    "layered_property_model",
    "acoustic_impedance",
    "generate_mine",
    "MineDataset",
    "unique_hole_xy_indices",
    "subset_holes",
]
