"""Interactive 3-D HTML exports via Plotly.

Each function writes a self-contained ``.html`` page (Plotly.js embedded) that
can be opened in any browser and rotated/zoomed/sliced interactively -- no
server required.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _flatten_grid(field, gx, gy, gz):
    vol = np.asarray(field, dtype=float)
    if vol.ndim != 3:
        raise ValueError("field must be 3-D (nx, ny, nz)")
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel(), vol.ravel()


def export_volume_html(
    field, gx, gy, gz, outfile, title="3D volume",
    colorscale="Viridis", colorbar_title="", opacity=0.12, surface_count=17,
    include_plotlyjs=True,
):
    """Interactive volume rendering (semi-transparent iso-shells)."""
    x, y, z, v = _flatten_grid(field, gx, gy, gz)
    fig = go.Figure(
        data=go.Volume(
            x=x, y=y, z=z, value=v,
            isomin=float(np.percentile(v, 5)),
            isomax=float(np.percentile(v, 95)),
            opacity=opacity, surface_count=surface_count,
            colorscale=colorscale, colorbar=dict(title=colorbar_title),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Elevation (m)"),
    )
    fig.write_html(outfile, include_plotlyjs=include_plotlyjs)
    return outfile


def export_isosurface_html(
    field, gx, gy, gz, outfile, title="3D isosurface",
    colorscale="Viridis", colorbar_title="", n_iso=5, opacity=0.6,
    include_plotlyjs=True,
):
    """Interactive isosurface rendering."""
    x, y, z, v = _flatten_grid(field, gx, gy, gz)
    fig = go.Figure(
        data=go.Isosurface(
            x=x, y=y, z=z, value=v,
            isomin=float(np.percentile(v, 20)),
            isomax=float(np.percentile(v, 80)),
            surface_count=n_iso, opacity=opacity,
            colorscale=colorscale, colorbar=dict(title=colorbar_title),
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Elevation (m)"),
    )
    fig.write_html(outfile, include_plotlyjs=include_plotlyjs)
    return outfile


def export_slices_html(
    field, gx, gy, gz, outfile, elevations_idx, title="3D slices",
    colorscale="Viridis", colorbar_title="", include_plotlyjs=True,
):
    """Interactive stack of horizontal slices at the given z indices."""
    vol = np.asarray(field, dtype=float)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    vmin, vmax = float(vol.min()), float(vol.max())
    fig = go.Figure()
    for k, zi in enumerate(elevations_idx):
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=np.full_like(xx, gz[zi]),
                surfacecolor=vol[:, :, zi],
                colorscale=colorscale, cmin=vmin, cmax=vmax,
                showscale=(k == 0), colorbar=dict(title=colorbar_title),
                name=f"z={gz[zi]:.0f} m",
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Elevation (m)"),
    )
    fig.write_html(outfile, include_plotlyjs=include_plotlyjs)
    return outfile
