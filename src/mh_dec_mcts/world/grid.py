from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid:
    size_x: float = 50.0      # metres
    size_y: float = 50.0
    cell: float = 0.125       # metres per cell
    centered: bool = True

    @property
    def shape(self) -> tuple[int, int]:          # (rows, cols)
        return (round(self.size_y / self.cell), round(self.size_x / self.cell))

    def xy_to_ij(self, x: float, y: float) -> tuple[int, int]:
        """
            Convert real-world (x, y) coordinates to grid indices (i, j).
        """
        if self.centered:
            center_i, center_j = (dim // 2 for dim in self.shape)
            j = x / self.cell + center_j
            i = -y / self.cell + center_i
        else:
            j = x / self.cell
            i = self.shape[0] - y / self.cell
        return int(np.round(i)), int(np.round(j))
     
    def ij_to_xy(self, i: int, j: int) -> tuple[float, float]:
        """
        Convert grid indices (i, j) to real-world (x, y) coordinates.
        """
        if self.centered:
            center_i, center_j = (dim // 2 for dim in self.shape)
            x = (j - center_j) * self.cell
            y = -(i - center_i) * self.cell
        else:
            x = j * self.cell
            y = (self.shape[0] - i) * self.cell
        return (x, y)
