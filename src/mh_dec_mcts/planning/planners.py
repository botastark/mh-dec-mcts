"""Planners, and the factory that builds any of them for a scenario.

Team interface (what run_single and run_multi call; one agent = a team of one):

    plan(beliefs, poses, nbrs) -> one action per agent
        beliefs, poses: per agent; nbrs[r]: who hears agent r at the current poses.

Planners (make_planner names):
    ig       greedy IG (paper Eq. 8-9), every agent on its own belief; config `greedy: {stride}`
             scores looks on a strided grid as Dec-MCTS does (1 = full grid, the paper's)
    igd      IG discounted by footprint overlap with the neighbours (paper Sec. III-E)
    random   uniformly random admissible action (paper baseline)
    sweep    climb to the optimal altitude, then lawnmower (paper baseline)
    decmcts  Dec-MCTS (planning/dec_mcts.py), config section `mcts`
    mhmcts   MH-MCTS (planning/mh_mcts.py), low level from `mcts`, the rest from `mh`
    mhgreedy MH with a greedy low level (mh_mcts.MHGreedy): IG / IGd as the setting's greedy plus
             g2; `mh` section, stride from `greedy`
Options of every planner: greedy.planner_seed / mcts.planner_seed (k > 0: other planning
randomness on the same runs); mcts.switch_cov (decmcts, mhmcts): PhaseSwitch to the setting's
greedy once the team's mean coverage reaches it.

ig / random / sweep are single-agent policies, choose(belief, pose, visited, rng) -> action
(`visited`: poses flown so far, oldest first, used only by Sweep), run per agent by Independent.
"""

import numpy as np

from mh_dec_mcts.comms.network import neighbours
from mh_dec_mcts.planning.dec_mcts import DecMCTS, DecMCTSConfig
from mh_dec_mcts.planning.information import info_gain
from mh_dec_mcts.planning.mh_mcts import MHConfig, MHGreedy, make_mh_mcts, planning_seed
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
    """IG_a = sum of per-cell IG over the footprint after action a (paper Eq. 8-9).

    stride > 1 ("sgreedy"): the sum runs over every stride-th cell of the global grid (the cells
    Dec-MCTS's objective.PlanGrid sees, same ceil bounds) and is scaled by stride**2."""

    def __init__(self, lat: Lattice, rates: SensorRates, tie_eps: float = 1e-4, stride: int = 1):
        self.lat, self.rates, self.tie_eps, self.stride = lat, rates, tie_eps, stride
        self.last_scores: dict[str, float] = {}

    def scores(self, belief: np.ndarray, pose: Pose) -> dict[str, float]:
        s = self.stride
        if s > 1:
            return self._strided_scores(belief[::s, ::s], pose, s)
        out = {}
        for a in self.lat.admissible(pose):
            q = self.lat.step(pose, a)
            i0, i1, j0, j1 = self.lat.footprint(q)
            out[a] = float(np.sum(info_gain(belief[i0:i1, j0:j1], *self.rates.at(q.k))))
        return out

    def _strided_scores(self, sub: np.ndarray, pose: Pose, s: int) -> dict[str, float]:
        out = {}
        for a in self.lat.admissible(pose):
            q = self.lat.step(pose, a)
            i0, i1, j0, j1 = (-(-v // s) for v in self.lat.footprint(q))  # ceil: the global sub-grid
            out[a] = float(np.sum(info_gain(sub[i0:i1, j0:j1], *self.rates.at(q.k)))) * s * s
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


class Independent:
    """A team of independent single-agent policies: agent a runs policies[a] on its own belief
    with its own generator rngs[a], and its own flown path as `visited`."""

    def __init__(self, policies: list, rngs: list[np.random.Generator]):
        self.policies, self.rngs = policies, rngs
        self.visited: list[list[Pose]] | None = None

    def plan(self, beliefs, poses, nbrs) -> list[str]:
        if self.visited is None:
            self.visited = [[p] for p in poses]
        else:
            for path, p in zip(self.visited, poses):
                path.append(p)
        return [pol.choose(b, p, v, rng)
                for pol, b, p, v, rng in zip(self.policies, beliefs, poses, self.visited, self.rngs)]


def iou(a, b) -> float:
    """Intersection over union of two footprints (i0, i1, j0, j1), half-open."""
    di = min(a[1], b[1]) - max(a[0], b[0])
    dj = min(a[3], b[3]) - max(a[2], b[2])
    if di < 0 or dj < 0:
        return 0.0
    inter = di * dj
    return inter / float((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter)


class IGd:
    """IG discounted (paper Sec. III-E, Pierdicca's simulate.py). After an IG pass, agents
    re-decide in a random order (order_rng); each candidate's IG is multiplied by (1 - IoU) with
    every neighbour's footprint at the neighbour's predicted pose (already-decided agents at
    their new pose). Neighbours within `range_` waypoint steps (None = everyone). One agent: IG."""

    def __init__(self, lat: Lattice, rates: SensorRates, rngs: list[np.random.Generator],
                 order_rng: np.random.Generator, range_: int | None, tie_eps: float = 1e-4, stride: int = 1):
        self.lat, self.g = lat, GreedyIG(lat, rates, tie_eps, stride)
        self.rngs, self.order_rng, self.range, self.tie_eps = rngs, order_rng, range_, tie_eps

    def plan(self, beliefs, poses, nbrs) -> list[str]:
        lat, n = self.lat, len(poses)
        scores = [self.g.scores(beliefs[a], poses[a]) for a in range(n)]
        actions = [pick_best(scores[a], self.rngs[a], self.tie_eps) for a in range(n)]  # IG pass
        if n == 1:
            return actions
        predicted = list(poses)
        actions = ["hover"] * n
        for a in self.order_rng.permutation(n):
            for nb in neighbours(predicted, self.range)[a]:
                nb_fp = lat.footprint(predicted[nb])
                for act in scores[a]:
                    scores[a][act] *= 1.0 - iou(nb_fp, lat.footprint(lat.step(poses[a], act)))
            actions[a] = pick_best(scores[a], self.rngs[a], self.tie_eps)
            predicted[a] = lat.step(poses[a], actions[a])
        return actions


class PhaseSwitch:
    """A tree planner until the team's mean coverage (share of cells each UAV's belief has seen,
    b != 0.5, averaged over the UAVs) reaches `cov`, then the greedy planner for the rest of the
    mission. switched_at: the first step planned by greedy (None: never)."""

    def __init__(self, first, then, cov: float):
        self.first, self.then, self.cov = first, then, cov
        self.t, self.switched_at = 0, None

    def plan(self, beliefs, poses, nbrs) -> list[str]:
        if self.switched_at is None and np.mean([np.mean(b != 0.5) for b in beliefs]) >= self.cov:
            self.switched_at = self.t
        self.t += 1
        return (self.first if self.switched_at is None else self.then).plan(beliefs, poses, nbrs)


PLANNERS = ("ig", "igd", "random", "sweep", "decmcts", "mhmcts", "mhgreedy")


def make_planner(name: str, cfg: dict, sc, range_: int | None = None, base: str | None = None):
    """Any planner for scenario sc (fresh state: build one per run). Randomness comes from the
    scenario's per-agent plan_rngs and order_rng (Dec-MCTS: its own, seeded from the run).
    range_: IGd's neighbour range, default comms.range. base: the setting's greedy (ig | igd),
    for mhgreedy and the phase switch (default ig). compat "pierdicca": exact-tie greedy IG
    and his sweep pattern."""
    from mh_dec_mcts.scenario import with_defaults

    cfg = with_defaults(cfg)
    pierdicca = cfg["compat"] == "pierdicca"
    tie_eps = 0.0 if pierdicca else 1e-4
    lat, rates, n = sc.lattice, sc.rates, len(sc.starts)
    stride = int(cfg["greedy"]["stride"])
    rng_range = range_ if range_ is not None else cfg["comms"]["range"]
    plan_rngs, order_rng = sc.plan_rngs, sc.order_rng
    if (k := int(cfg["greedy"].get("planner_seed", 0))) and name in ("ig", "igd", "mhgreedy"):
        plan_rngs = [np.random.default_rng([sc.run_seed, 5, k, i]) for i in range(n)]  # other tie-breaks
        order_rng = np.random.default_rng([sc.run_seed, 6, k])  # and IGd decision orders
    switch = cfg["mcts"].get("switch_cov")
    if switch is not None and name in ("decmcts", "mhmcts"):
        first = make_planner(name, {**cfg, "mcts": {**cfg["mcts"], "switch_cov": None}}, sc, range_, base)
        return PhaseSwitch(first, make_planner(base or "ig", cfg, sc, range_), float(switch))
    if name == "ig":
        g = GreedyIG(lat, rates, tie_eps, stride)  # stateless: one for the whole team
        return Independent([g] * n, plan_rngs)
    if name == "igd":
        return IGd(lat, rates, plan_rngs, order_rng, rng_range, tie_eps, stride)
    if name == "mhgreedy":
        return MHGreedy(lat, rates, MHConfig(**cfg["mh"]), plan_rngs, order_rng, rng_range,
                        (base or "ig") == "igd", int(cfg["run"]["steps"]), stride, tie_eps,
                        planning_seed(sc.run_seed, int(cfg["greedy"].get("planner_seed", 0))))
    if name == "random":
        return Independent([RandomPlanner(lat)] * n, sc.plan_rngs)
    if name == "sweep":
        return Independent([PierdiccaSweep(lat) if pierdicca else Sweep(lat) for _ in range(n)], sc.plan_rngs)
    if name == "decmcts":
        c = DecMCTSConfig(**cfg["mcts"])
        return DecMCTS(lat, rates, c, planning_seed(sc.run_seed, c.planner_seed))
    if name == "mhmcts":
        return make_mh_mcts(cfg, sc)
    raise ValueError(f"unknown planner '{name}' (known: {', '.join(PLANNERS)})")
