"""Single-step planners: greedy IG (paper), Random and Sweep baselines.

Every planner has  choose(belief, pose, visited, rng) -> action.
`visited` is the list of poses flown so far (oldest first), used only by Sweep.
"""

import numpy as np

from mh_dec_mcts.planning.information import info_gain
from mh_dec_mcts.sensing.lattice import ACTIONS, Lattice, Pose
from mh_dec_mcts.sensing.sensor import SensorRates


def pick_best(scores: dict[str, float], rng: np.random.Generator, eps: float = 1e-4) -> str:
    """argmax (paper Eq. 9); actions within eps of the best are tied and chosen uniformly at
    random (legacy planner.ig_based). Candidates are listed in fixed ACTIONS order.
    eps=0: exact ties only (Pierdicca's Planner._non_targeted_mini)."""
    best = max(scores.values())
    tied = [a for a in ACTIONS if a in scores and scores[a] >= best - eps]
    return tied[rng.integers(len(tied))] if len(tied) > 1 else tied[0]


class GreedyIG:
    """IG_a = sum of per-cell IG over the footprint after action a (paper Eq. 8-9)."""

    def __init__(self, lat: Lattice, rates: SensorRates, tie_eps: float = 1e-4):
        self.lat, self.rates, self.tie_eps = lat, rates, tie_eps
        self.last_scores: dict[str, float] = {}

    def scores(self, belief: np.ndarray, pose: Pose) -> dict[str, float]:
        out = {}
        for a in self.lat.admissible(pose):
            q = self.lat.move(pose, a)
            assert q is not None  # admissible() only returns moves that stay on the lattice
            i0, i1, j0, j1 = self.lat.footprint(q)
            out[a] = float(np.sum(info_gain(belief[i0:i1, j0:j1], *self.rates.at(q.k))))
        return out

    def choose(self, belief, pose, visited, rng) -> str:
        self.last_scores = self.scores(belief, pose)
        return pick_best(self.last_scores, rng, self.tie_eps)


class RandomPlanner:
    """Paper baseline: next action uniformly at random among the admissible ones."""

    def __init__(self, lat: Lattice):
        self.lat = lat

    def choose(self, belief, pose, visited, rng) -> str:
        ok = self.lat.admissible(pose)
        return ok[rng.integers(len(ok))]


class Sweep:
    """Paper baseline (legacy planner.sweep): climb to the optimal altitude, then lawnmower.

    Moves to unvisited waypoints only; the row direction (LeftRight / BackFront) is fixed at
    the first decision, randomly unless only one direction is possible.
    """

    ORDER = {"LeftRight": ("left", "right", "front", "back"),
             "BackFront": ("back", "front", "left", "right")}

    def __init__(self, lat: Lattice, optimal_altitude: float = 21.65):
        self.lat = lat
        self.k_opt = int(np.searchsorted(lat.altitudes, optimal_altitude - 1e-6))
        self.direction: str | None = None

    def choose(self, belief, pose, visited, rng) -> str:
        ok = self.lat.admissible(pose)
        if pose.k < self.k_opt and "up" in ok:
            return "up"
        seen = set(visited)
        free = [a for a in ok if a not in ("up", "down", "hover") and self.lat.move(pose, a) not in seen]
        direction = self.direction
        if direction is None:
            if len(free) == 1:
                direction = "LeftRight" if free[0] in ("left", "right") else "BackFront"
            else:
                direction = ("LeftRight", "BackFront")[rng.integers(2)]
            self.direction = direction
        for a in self.ORDER[direction]:
            if a in free:
                return a
        return "hover"


class PierdiccaSweep:
    """Pierdicca's Planner._sweep for one agent: climb to the optimal altitude, then columns
    along y (front to the north edge, one step sideways, back to the south edge, ...).

    The sideways direction starts as "right" and flips every n_x * n_y decisions (counting
    climbing steps), exactly as in the original. A move that would leave the lattice was
    clipped to the edge there, i.e. the UAV stayed put: here it returns "hover" instead,
    while remembering the intended action for the next decision.
    """

    def __init__(self, lat: Lattice, optimal_altitude: float = 21.65):
        self.lat = lat
        self.k_opt = int(np.searchsorted(lat.altitudes, optimal_altitude - 1e-6))
        self.n_tot = lat.nx * lat.ny
        self.side, self.last, self.n_visited = "left", "up", 0

    def choose(self, belief, pose, visited, rng) -> str:
        if self.n_visited % self.n_tot == 0:
            self.side = "right" if self.side == "left" else "left"
        top, bottom = pose.iy == self.lat.ny - 1, pose.iy == 0
        last = self.last
        if last == "up":
            a = "up" if pose.k != self.k_opt else "front"
        elif last == "front":
            a = self.side if top else "front"
        elif last == "back":
            a = self.side if bottom else "back"
        else:  # last was a sideways step
            a = "back" if top else "front"
        self.last = a
        self.n_visited += 1
        return a if self.lat.move(pose, a) is not None else "hover"


def make_planner(name: str, lat: Lattice, rates: SensorRates, compat: str | None = None):
    """compat="pierdicca": exact-tie greedy IG and Pierdicca's sweep pattern.
    Dec-MCTS needs the config and scenario: planning.dec_mcts.make_dec_mcts(cfg, sc)."""
    pierdicca = compat == "pierdicca"
    if name == "ig":
        return GreedyIG(lat, rates, tie_eps=0.0 if pierdicca else 1e-4)
    if name == "random":
        return RandomPlanner(lat)
    if name == "sweep":
        return PierdiccaSweep(lat) if pierdicca else Sweep(lat)
    raise ValueError(f"unknown planner '{name}'")
