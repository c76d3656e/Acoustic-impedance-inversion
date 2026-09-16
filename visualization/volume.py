"""Optional 3-D volume rendering via PyVista (off-screen).

3-D rendering needs a GPU or an OSMesa/Xvfb software stack that is not always
present.  ``render_volume`` therefore degrades gracefully: it attempts an
off-screen PyVista render and returns ``None`` (without raising) if the runtime
cannot render, so the end-to-end pipeline never fails on visualisation alone.
"""

from __future__ import annotations

import numpy as np


def render_volume(volume: np.ndarray, outfile: str, z_index=None):
    """Render orthogonal slices of an impedance volume to a PNG.

    Returns the output path on success, or ``None`` if PyVista/off-screen
    rendering is unavailable.
    """
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        vol = np.asarray(volume, dtype=float)
        grid = pv.ImageData(dimensions=np.array(vol.shape) + 1)
        grid.cell_data["AI"] = vol.flatten(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
        plotter.add_mesh_clip_plane(
            grid, scalars="AI", cmap="viridis",
            scalar_bar_args={"title": "Acoustic Impedance"},
        )
        plotter.add_axes()
        plotter.show(screenshot=outfile)
        plotter.close()
        return outfile
    except Exception as exc:  # noqa: BLE001 - visualisation must never be fatal
        print(f"[visualization] 3-D render skipped: {exc}")
        return None


def render_isosurface(volume, outfile, n_contours=6, cmap="viridis"):
    """Render isosurfaces of a volume to a PNG (off-screen).

    Returns the output path on success, or ``None`` if PyVista/off-screen
    rendering is unavailable.
    """
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        vol = np.asarray(volume, dtype=float)
        grid = pv.ImageData(dimensions=vol.shape)
        grid.point_data["value"] = vol.flatten(order="F")
        lo, hi = np.percentile(vol, [20, 80])
        contours = grid.contour(np.linspace(lo, hi, n_contours))

        plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
        plotter.add_mesh(contours, cmap=cmap, opacity=0.55,
                         scalar_bar_args={"title": "value"})
        plotter.add_axes()
        plotter.show(screenshot=outfile)
        plotter.close()
        return outfile
    except Exception as exc:  # noqa: BLE001
        print(f"[visualization] isosurface render skipped: {exc}")
        return None
