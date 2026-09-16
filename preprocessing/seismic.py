"""SEG-Y read/write helpers built on ``segyio``."""

from __future__ import annotations

import numpy as np
import segyio


def write_segy(path: str, cube: np.ndarray, dt_us: int = 2000) -> str:
    """Write a 3-D cube ``(n_iline, n_xline, n_samples)`` to a SEG-Y file.

    ``dt_us`` is the sample interval in microseconds (2000 us = 2 ms).
    """
    data = np.asarray(cube, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("cube must be 3-D (n_iline, n_xline, n_samples)")
    ni, nx, nt = data.shape

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
    spec.samples = np.arange(nt)
    spec.ilines = np.arange(1, ni + 1)
    spec.xlines = np.arange(1, nx + 1)

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        tr = 0
        for il in range(ni):
            for xl in range(nx):
                f.header[tr] = {
                    segyio.su.iline: il + 1,
                    segyio.su.xline: xl + 1,
                    segyio.su.dt: dt_us,
                }
                f.trace[tr] = data[il, xl, :]
                tr += 1
    return path


def read_segy(path: str):
    """Read a structured SEG-Y file into ``(cube, dt_seconds)``."""
    with segyio.open(path, "r") as f:
        cube = segyio.tools.cube(f)
        dt = segyio.tools.dt(f) / 1.0e6  # microseconds -> seconds
    return np.asarray(cube, dtype=float), float(dt)
