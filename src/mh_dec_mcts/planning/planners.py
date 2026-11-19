"""Single-step planners: greedy IG (paper), Random and Sweep baselines.
 
Every planner has  choose(belief, pose, visited, rng) -> action.
`visited` is the list of poses flown so far (oldest first), used only by Sweep.
"""
 
import numpy as np
 
from mh_dec_mcts.planning.information import info_gain
from mh_dec_mcts.sensing.lattice import ACTIONS, Lattice, Pose
from mh_dec_mcts.sensing.sensor import SensorRates

def pick_best(scores: dict[str, float], rng: np.random.Generator, eps: float = 1e-4) -> str:
    """argmax IG (paper Eq. 9); ties within eps broken uniformly at random (legacy planner.ig_based)."""
    best = max(scores.values())
    tied = [a for a in ACTIONS if a in scores and scores[a] >= best - eps]
    return rng.choice(tied)

class GreedyIG:
    """IG_a = sum of per-cell IG over the footprint after action a (paper Eq. 8-9)."""
 
    def __init__(self, lat: Lattice, rates: SensorRates):
        self.lat, self.rates = lat, rates
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
        return pick_best(self.last_scores, rng)

 
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
 
 
def make_planner(name: str, lat: Lattice, rates: SensorRates):
    if name == "ig":
        return GreedyIG(lat, rates)
    if name == "random":
        return RandomPlanner(lat)
    if name == "sweep":
        return Sweep(lat)
    raise ValueError(f"unknown planner '{name}'")
 