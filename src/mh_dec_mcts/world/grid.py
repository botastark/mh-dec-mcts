"""Field grid geometry: shape and conversions between metres (x, y) and cells (i, j).

Conventions (identical to legacy ``uav_camera.Camera``):
  - i is the row index and grows southwards (-y); j is the column index and grows eastwards (+x).
  - ``centered=True``: (x, y) = (0, 0) is the cell (rows // 2, cols // 2).
  - ``centered=False``: (x, y) = (0, 0) is the bottom-left corner, i.e. cell (rows, 0).
  - ``xy_to_ij`` rounds half to even (``np.rint``), exactly like legacy ``int(np.round(.))``.
    Results may equal ``rows``/``cols`` on the far edge: they are valid slice ends but not
    valid cell indices; use ``in_bounds`` / ``clip_ij`` when you need a cell.

All conversions accept scalars or numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Grid:
    size_x: float = 50.0  # field width along x, metres
    size_y: float = 50.0  # field height along y, metres
    cell: float = 0.125  # cell side, metres
    centered: bool = True

    def __post_init__(self) -> None:
        if self.size_x <= 0 or self.size_y <= 0 or self.cell <= 0:
            raise ValueError(f"Grid sizes must be positive, got {self}")
        for name, size in (("size_x", self.size_x), ("size_y", self.size_y)):
            n = size / self.cell
            if not np.isclose(n, round(n)):
                raise ValueError(f"{name}={size} is not a whole number of cells of {self.cell}")

    # ------------------------------------------------------------------ shape
    @property
    def rows(self) -> int:
        return round(self.size_y / self.cell)

    @property
    def cols(self) -> int:
        return round(self.size_x / self.cell)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols), the numpy shape of any map on this grid."""
        return (self.rows, self.cols)

    @property
    def center_ij(self) -> tuple[int, int]:
        return (self.rows // 2, self.cols // 2)

    @property
    def x_range(self) -> tuple[float, float]:
        """(x_min, x_max) of the field in metres."""
        if self.centered:
            return (-self.center_ij[1] * self.cell, (self.cols - self.center_ij[1]) * self.cell)
        return (0.0, self.cols * self.cell)

    @property
    def y_range(self) -> tuple[float, float]:
        """(y_min, y_max) of the field in metres."""
        if self.centered:
            return (-(self.rows - self.center_ij[0]) * self.cell, self.center_ij[0] * self.cell)
        return (0.0, self.rows * self.cell)

    # ------------------------------------------------------------ conversions


    def ij_to_xy(self, i, j):
        """
        Convert grid indices (i, j) to real-world (x, y) coordinates.
        """

        if self.centered:
            ci, cj = self.center_ij
            x = (j - cj) * self.cell
            y = -(i - ci) * self.cell
        else:
            x = j * self.cell
            y = (self.rows - i) * self.cell
        return (x, y)

    def xy_to_ij(self, x, y):
        """
        Convert real-world (x, y) coordinates to grid indices (i, j).

        Parameters:
            x (float): X coordinate.
            y (float): Y coordinate.
        Returns:
            tuple: (i, j) grid indices as integers.
        """
        if self.centered:
            ci, cj = self.center_ij
            j = x / self.cell + cj
            i = -y / self.cell + ci
        else:
            j = x / self.cell
            i = self.rows - y / self.cell
        return int(np.round(i)), int(np.round(j))


    # ----------------------------------------------------------------- bounds
    def in_bounds(self, i: ArrayLike, j: ArrayLike):
        """True where (i, j) is a valid cell index."""
        i = np.asarray(i)
        j = np.asarray(j)
        ok = (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
        return bool(ok) if ok.ndim == 0 else ok

    def clip_ij(self, i: ArrayLike, j: ArrayLike):
        """Clamp (i, j) to the nearest valid cell index."""
        i = np.clip(np.asarray(i), 0, self.rows - 1)
        j = np.clip(np.asarray(j), 0, self.cols - 1)
        if i.ndim == 0:
            return int(i), int(j)
        return i.astype(int), j.astype(int)

    def clip_bounds(self, imin: int, imax: int, jmin: int, jmax: int) -> tuple[int, int, int, int]:
        """Clamp a half-open box [imin, imax) x [jmin, jmax) to the grid (slice semantics)."""
        imin, imax = max(0, min(imin, self.rows)), max(0, min(imax, self.rows))
        jmin, jmax = max(0, min(jmin, self.cols)), max(0, min(jmax, self.cols))
        return imin, imax, jmin, jmax

    def contains_xy(self, x: ArrayLike, y: ArrayLike):
        """True where the metric point lies inside the field (edges included)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        ok = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return bool(ok) if ok.ndim == 0 else ok