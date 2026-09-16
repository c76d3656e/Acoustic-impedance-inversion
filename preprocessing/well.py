"""LAS well-log helpers built on ``lasio``.

Computes well-side acoustic impedance from sonic (DT) and density (RHOB) logs:

    Vp = 1e6 / DT            (DT in us/m -> Vp in m/s)
    AI = Vp * rho
"""

from __future__ import annotations

import numpy as np
import lasio


def read_las(path: str) -> lasio.LASFile:
    """Read a LAS file."""
    return lasio.read(path)


def well_acoustic_impedance(
    las: lasio.LASFile,
    dt_curve: str = "DT",
    rho_curve: str = "RHOB",
    dt_unit: str = "us/m",
    rho_unit: str = "g/cc",
):
    """Return ``(depth, acoustic_impedance)`` arrays from a LAS file."""
    df = las.df()
    depth = df.index.to_numpy(dtype=float)
    dt = df[dt_curve].to_numpy(dtype=float)
    rho = df[rho_curve].to_numpy(dtype=float)

    vp = 1.0e6 / dt
    if dt_unit == "us/ft":
        vp *= 0.3048  # ft/s -> m/s

    rho_si = rho * 1000.0 if rho_unit == "g/cc" else rho
    return depth, vp * rho_si


def write_synthetic_las(
    path: str,
    depth: np.ndarray,
    vp: np.ndarray,
    rho: np.ndarray,
    well_name: str = "SYNTH-1",
) -> str:
    """Write a minimal LAS file with DEPTH/DT/RHOB curves (for tests/examples)."""
    depth = np.asarray(depth, dtype=float)
    vp = np.asarray(vp, dtype=float)
    rho = np.asarray(rho, dtype=float)

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value=well_name)
    las.append_curve("DEPT", depth, unit="m")
    las.append_curve("DT", 1.0e6 / vp, unit="us/m", descr="Sonic slowness")
    las.append_curve("RHOB", rho / 1000.0, unit="g/cc", descr="Bulk density")
    las.write(path, version=2.0)
    return path
