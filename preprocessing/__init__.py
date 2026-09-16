"""Field-data preprocessing: SEG-Y seismic, LAS well logs, depth<->time.

These helpers back the Stage II (Penobscot 3D) workflow.  They are intentionally
light wrappers around ``segyio`` and ``lasio`` so the same inversion engine used
for the synthetic Stage I model can be applied to real data.
"""

from .seismic import write_segy, read_segy
from .well import read_las, well_acoustic_impedance, write_synthetic_las
from .depth_time import depth_to_twt, resample_to_time

__all__ = [
    "write_segy",
    "read_segy",
    "read_las",
    "well_acoustic_impedance",
    "write_synthetic_las",
    "depth_to_twt",
    "resample_to_time",
]
