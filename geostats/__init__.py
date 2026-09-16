"""3-D geostatistics: kriging of point estimates into continuous fields.

Both estimators return a mean field **and** a variance field so the fusion step
can weight each data source by its local reliability.
"""

from .kriging import (
    ordinary_kriging_3d,
    regression_kriging_3d,
    make_grid,
)

__all__ = ["ordinary_kriging_3d", "regression_kriging_3d", "make_grid"]
