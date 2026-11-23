from dataclasses import dataclass
import math

import numpy as np

from mh_dec_mcts.world.grid import Grid


@dataclass(frozen=True)
class Pose:
    ix: int  # waypoint index along x (0 = west edge)
    iy: int  # waypoint index along y (0 = south edge)
    k: int  # altitude level (0 = lowest)


ACTIONS = ("up", "down", "front", "back", "right", "left", "hover")  # iteration order

_DELTA = {  # (dix, diy, dk); front = +y, right = +x (legacy convention)
    "up": (0, 0, 1),
    "down": (0, 0, -1),
    "front": (0, 1, 0),
    "back": (0, -1, 0),
    "right": (1, 0, 0),
    "left": (-1, 0, 0),
    "hover": (0, 0, 0),
}


@dataclass(frozen=True)
class Lattice:
    grid: Grid
    xy_step: float  # metres between waypoints
    h_min: float  # lowest altitude, metres
    h_step: float  # metres between altitude levels
    n_alt: int = 6
    fov_deg: float = 60.0

    def __post_init__(self):
        if self.xy_step <= 0 or self.h_step <= 0 or self.h_min <= 0 or self.n_alt < 1:
            raise ValueError(f"invalid lattice: {self}")
        if self.nx < 2 or self.ny < 2:
            raise ValueError(f"xy_step {self.xy_step} too large for the field")

    # field edges in metres
    @property
    def x0(self) -> float:
        return -self.grid.size_x / 2 if self.grid.centered else 0.0

    @property
    def y0(self) -> float:
        return -self.grid.size_y / 2 if self.grid.centered else 0.0

    @property
    def nx(self) -> int:
        return math.floor(self.grid.size_x / self.xy_step + 1e-9) + 1

    @property
    def ny(self) -> int:
        return math.floor(self.grid.size_y / self.xy_step + 1e-9) + 1

    @property
    def altitudes(self) -> np.ndarray:
        return self.h_min + self.h_step * np.arange(self.n_alt)

    def xyz(self, p: Pose) -> tuple[float, float, float]:
        return (
            self.x0 + p.ix * self.xy_step,
            self.y0 + p.iy * self.xy_step,
            self.h_min + p.k * self.h_step,
        )

    def move(self, p: Pose, action: str) -> Pose | None:
        """Next pose, or None if the action would leave the lattice (hover is always allowed)."""
        dx, dy, dk = _DELTA[action]  # KeyError on unknown action
        q = Pose(p.ix + dx, p.iy + dy, p.k + dk)
        if 0 <= q.ix < self.nx and 0 <= q.iy < self.ny and 0 <= q.k < self.n_alt:
            return q
        return None

    def step(self, p: Pose, action: str) -> Pose:
        """move() for an action known to be admissible: raises ValueError instead of returning None."""
        q = self.move(p, action)
        if q is None:
            raise ValueError(f"inadmissible action {action!r} at {p}")
        return q

    def admissible(self, p: Pose) -> list[str]:
        return [a for a in ACTIONS if self.move(p, a) is not None]

    def footprint(self, p: Pose) -> tuple[int, int, int, int]:
        """Cells seen from p as (imin, imax, jmin, jmax), half-open, clipped to the field."""
        x, y, h = self.xyz(p)
        c = self.grid.cell
        half = round(h * math.tan(math.radians(self.fov_deg) / 2) / c) * c
        x_lo, x_hi = max(x - half, self.x0), min(x + half, self.x0 + self.grid.size_x)
        y_lo, y_hi = max(y - half, self.y0), min(y + half, self.y0 + self.grid.size_y)
        imax, jmin = self.grid.xy_to_ij(x_lo, y_lo)  # rows grow southwards
        imin, jmax = self.grid.xy_to_ij(x_hi, y_hi)
        return imin, imax, jmin, jmax


def start_poses(lat: Lattice, n_agents: int, rng: np.random.Generator | None = None) -> list[Pose]:
    """Legacy "corner" start, lowest altitude.

    1 agent:  a random corner, drawn as rng.integers(4) over [SW, NW, SE, NE]
              (legacy single-agent branch; pass np.random.default_rng(run_seed)).
    N agents: fixed order SW, SE, NW, NE, then edge midpoints S, N, W, E repeating
              (legacy MultiAgentCoordinator.reset_start_position; rng unused).
    """
    e, n = lat.nx - 1, lat.ny - 1
    if n_agents == 1:
        if rng is None:
            raise ValueError("single-agent corner start needs an rng")
        ix, iy = [(0, 0), (0, n), (e, 0), (e, n)][rng.integers(4)]
        return [Pose(ix, iy, 0)]
    if (e % 2 or n % 2) and n_agents > 4:
        raise ValueError("edge-midpoint starts need an odd number of waypoints per axis")
    corners = [(0, 0), (e, 0), (0, n), (e, n)]
    mids = [(e // 2, 0), (e // 2, n), (0, n // 2), (e, n // 2)]
    xy = corners[:n_agents] + [mids[i % 4] for i in range(max(0, n_agents - 4))]
    return [Pose(ix, iy, 0) for ix, iy in xy]

