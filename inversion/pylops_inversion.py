"""Post-stack seismic inversion via PyLops (alternative to the in-house engine).

Wraps ``pylops.avo.poststack.PoststackInversion`` so the seismic branch can use
a well-tested linear-operator inversion framework.  Works on log-impedance in
the sample domain, consistent with :mod:`inversion.forward`.
"""

from __future__ import annotations

import numpy as np
from pylops.avo.poststack import PoststackInversion


def poststack_inversion(
    seismic: np.ndarray,
    wavelet: np.ndarray,
    background: np.ndarray,
    epsR: float = 5.0,
    epsI: float = 1e-3,
):
    """Invert a 3-D seismic volume for acoustic impedance with PyLops.

    Parameters
    ----------
    seismic:
        ``(nx, ny, nz)`` seismic (depth/time on the last axis).
    wavelet:
        Source wavelet.
    background:
        ``(nx, ny, nz)`` low-frequency background impedance model.
    epsR:
        Spatial regularisation weight.
    epsI:
        Damping weight.

    Returns
    -------
    numpy.ndarray
        Inverted acoustic-impedance volume ``(nx, ny, nz)``.
    """
    d = np.asarray(seismic, dtype=float)
    if d.ndim != 3:
        raise ValueError("seismic must be 3-D (nx, ny, nz)")

    # PyLops expects the vertical axis first: (nz, nx, ny).
    d_t = np.moveaxis(d, -1, 0)
    m0 = np.log(np.moveaxis(np.asarray(background, dtype=float), -1, 0))

    inv, _ = PoststackInversion(
        d_t, wavelet / np.abs(wavelet).max(), m0=m0,
        explicit=False, epsR=epsR, epsI=epsI, simultaneous=False,
    )
    return np.exp(np.moveaxis(inv, 0, -1))
