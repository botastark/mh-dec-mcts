"""MH-MCTS (Kent et al., "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo
Tree Search", 2024) for the lattice mapping problem, built step by step.

Step 1: Regions, the square tiles the long-horizon part of the planner reasons about.
Step 2: MissionTime, the paper's estimate of the mission time (g2, Alg. 2) on these regions.
Step 3: MHMCTS, the low-level planner: Dec-MCTS over lattice moves (short horizon, `mcts`
        section, single-intent variant as in the paper) whose reward adds g2 (eq. 6-8):

            f = g1(sequence after the teammates' plans)
                + w * lam * (T[own plan = stay at the current pose] - T[own plan = tree part])

        g1 is the entropy drop Dec-MCTS already maximises. T is MissionTime: teammates enter it
        through their shared plans (length, end waypoint). Only the part of the sequence inside
        the tree enters g2; the rollout part is ignored (paper Sec. VI). lam converts the steps
        saved into bits. Every UAV takes the nearest open region next: a fixed stand-in for
        the high-level planner of step 5. With no open region left, g2 = 0 and the planner is
        plain Dec-MCTS with the short horizon; w = 0 gives Dec-MCTS (single variant) as well.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from mh_dec_mcts.planning.dec_mcts import DecMCTS, DecMCTSConfig, Node, Tree
from mh_dec_mcts.planning.information import binary_entropy
from mh_dec_mcts.planning.objective import PlanGrid
from mh_dec_mcts.sensing.lattice import Lattice, Pose
from mh_dec_mcts.sensing.sensor import SensorRates


Robot = tuple[int, tuple[int, int]]  # a UAV in MissionTime: (steps already planned, waypoint (ix, iy))


@dataclass
class MHConfig:
    tile: float = 25.0  # region side, metres (100 x 50 m field -> 4 x 2 regions)
    done_entropy: float = 0.3  # a region is finished below this mean cell entropy (bits; prior = 1)
    rate: float | str = "auto"  # lam, entropy drop per step in bits ("auto": look_rate at level 0)
    g2_weight: float = 1.0  # w in f = g1 + w lam (T_stay - T_plan); 0 = Dec-MCTS


class Regions:
    """Square tiles of the field, seen two ways:

    on the planning sub-grid (every stride-th cell, as PlanGrid): `label` gives each cell its
        region, so the entropy of a belief per region is one bincount;
    on the lattice: `box` gives each region's waypoint bounds, so the travel distance from a
        waypoint to a region is a Manhattan distance in waypoint steps.

    Region id = ry * nrx + rx, rx counted eastwards from the west edge, ry northwards from the
    south edge. Tiles are `tile` metres (rounded to whole waypoint steps); the last row and
    column take what is left of the field.
    """

    def __init__(self, lat: Lattice, stride: int, tile: float):
        g = lat.grid
        self.per = per = max(1, round(tile / lat.xy_step))  # waypoint steps per tile side
        self.nrx = max(1, math.ceil((lat.nx - 1) / per))
        self.nry = max(1, math.ceil((lat.ny - 1) / per))
        self.n = self.nrx * self.nry
        rows = np.arange(0, g.rows, stride)
        cols = np.arange(0, g.cols, stride)
        x, _ = g.ij_to_xy(np.zeros_like(cols), cols)
        _, y = g.ij_to_xy(rows, np.zeros_like(rows))
        rx = np.clip(((x - lat.x0) / (per * lat.xy_step)).astype(int), 0, self.nrx - 1)
        ry = np.clip(((y - lat.y0) / (per * lat.xy_step)).astype(int), 0, self.nry - 1)
        self.label = (ry[:, None] * self.nrx + rx[None, :]).ravel()
        self.cells = np.bincount(self.label, minlength=self.n).astype(float)  # sub-grid cells
        self.scale = float(stride * stride)  # full-resolution cells per sub-grid cell
        self.box = []  # (ix_lo, ix_hi, iy_lo, iy_hi): waypoint bounds, inclusive
        for r in range(self.n):
            ax, ay = r % self.nrx, r // self.nrx
            self.box.append((ax * per, min((ax + 1) * per, lat.nx - 1),
                             ay * per, min((ay + 1) * per, lat.ny - 1)))
        self.center = [((a + b) // 2, (c + d) // 2) for a, b, c, d in self.box]

    def entropy(self, b0: np.ndarray) -> np.ndarray:
        """Entropy (bits, full resolution) of each region of a sub-grid belief."""
        return np.bincount(self.label, weights=binary_entropy(b0).ravel(), minlength=self.n) * self.scale

    def mean_entropy(self, b0: np.ndarray) -> np.ndarray:
        """Mean cell entropy (bits) of each region; 1 = unknown, 0 = certain."""
        return self.entropy(b0) / (self.cells * self.scale)

    def dist(self, xy: tuple[int, int], r: int) -> int:
        """Waypoint steps from waypoint xy = (ix, iy) to the nearest waypoint of region r."""
        a, b, c, d = self.box[r]
        return max(0, a - xy[0], xy[0] - b) + max(0, c - xy[1], xy[1] - d)


def look_rate(lat: Lattice, rates: SensorRates, stride: int, level: int = 0) -> float:
    """Bits removed by one look at altitude `level` over unknown ground (belief 0.5): the
    nominal entropy drop per step that converts entropy into steps of work."""
    g = PlanGrid(np.full(lat.grid.shape, 0.5), lat, rates, stride)
    return g.gain(g.new(np.random.default_rng(0)), Pose(lat.nx // 2, lat.ny // 2, level))


class MissionTime:
    """g2 of the paper (Alg. 2): an estimate of the time, in steps, to finish the open regions.

    A region is open while its mean cell entropy is above `done`. Every UAV starts where its
    low-level plan ends, at time = the plan's length. Then, repeatedly, the UAV with the
    earliest time takes the nearest open region and spends
        travel  waypoint steps to the region (Regions.dist), plus
        work    the region's entropy above the closing level, divided by `lam` (bits/step),
    on it, and continues from the region's centre. The estimate is the mean of the UAVs'
    finishing times, as in the paper (a UAV that gets no region finishes at its start time).
    """

    def __init__(self, reg: Regions, lam: float, done: float):
        self.reg, self.lam, self.done = reg, lam, done

    def open(self, b0: np.ndarray) -> frozenset[int]:
        """Regions not finished yet in the sub-grid belief b0."""
        return frozenset(int(r) for r in np.flatnonzero(self.reg.mean_entropy(b0) > self.done))

    def work(self, E: np.ndarray, r: int) -> float:
        """Steps to bring region r (entropies E, bits) down to the closing level."""
        return max(0.0, E[r] - self.done * self.reg.cells[r] * self.reg.scale) / self.lam

    def __call__(self, E: np.ndarray, open_: frozenset[int], robots: list[Robot]) -> float:
        """robots: [(t0, (ix, iy))], each UAV's time and waypoint at the end of its plan."""
        rem = set(open_)
        t = [float(t0) for t0, _ in robots]
        pos = [xy for _, xy in robots]
        while rem:
            i = int(np.argmin(t))
            nxt = min(sorted(rem), key=lambda r: self.reg.dist(pos[i], r))
            t[i] += self.reg.dist(pos[i], nxt) + self.work(E, nxt)
            pos[i] = self.reg.center[nxt]
            rem.discard(nxt)
        return float(np.mean(t))


class RangeTree(Tree):
    """Dec-MCTS tree whose UCB scales the discounted means by the range [f_min, f_max] of the
    rewards seen instead of dividing by f_max: with g2 a reward can be negative."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.f_min, self.f_max = math.inf, -math.inf

    def backprop(self, node: Node, f: float) -> None:
        self.f_min = min(self.f_min, f)
        super().backprop(node, f)

    def score(self, mean: float) -> float:
        return (mean - self.f_min) / max(self.f_max - self.f_min, 1e-12)


class MHMCTS(DecMCTS):
    """Dec-MCTS, single-intent variant, whose rollouts add g2 to the reward of their tree part.
    Interface as DecMCTS: plan(beliefs, poses, nbrs) -> actions, choose(belief, pose)."""

    def __init__(self, lat: Lattice, rates: SensorRates, cfg: DecMCTSConfig | None = None,
                 mh: MHConfig | None = None, seed: int | Sequence[int] = 0):
        cfg = cfg or DecMCTSConfig()
        cfg.variant = "single"
        super().__init__(lat, rates, cfg, seed)
        self.mh = mh or MHConfig()
        self.reg = Regions(lat, cfg.stride, self.mh.tile)
        lam = look_rate(lat, rates, cfg.stride) if self.mh.rate == "auto" else float(self.mh.rate)
        self.T = MissionTime(self.reg, lam, self.mh.done_entropy)
        self.last_open: list[frozenset[int]] = []  # per UAV, regions open at the last plan()

    def _start_plan(self, grids) -> None:
        self.last_open = [self.T.open(g.base) for g in grids]  # each UAV judges with its own belief

    def _new_tree(self, root: Pose) -> Tree:
        return (RangeTree if self.mh.g2_weight else Tree)(self.lat, root, self.cfg)  # w = 0: Dec-MCTS

    def _rollout_context(self, r: int, known: dict) -> tuple[frozenset[int], list[Robot]]:
        """Agent r's open regions, and its teammates' last shared plans as (length, end waypoint)."""
        others = [(len(s[0]), (s[0][-1].ix, s[0][-1].iy)) for _, (s, _) in sorted(known.items()) if s[0]]
        return self.last_open[r], others

    def _tree_reward(self, tree, grid, g, seq, ctx) -> float:
        """g1 of the tree part plus w lam (T[stay] - T[tree part]); the rollout part gets g1 only."""
        open_, others = ctx
        if not open_ or self.mh.g2_weight == 0:
            return super()._tree_reward(tree, grid, g, seq, ctx)
        pose = tree.root.pose
        e_stay = self.reg.entropy(g.b)  # after the teammates' looks, before own
        f = super()._tree_reward(tree, grid, g, seq, ctx)
        e_plan = self.reg.entropy(g.b) if seq else e_stay
        end = seq[-1] if seq else pose
        t_stay = self.T(e_stay, open_, [(0, (pose.ix, pose.iy))] + others)
        t_plan = self.T(e_plan, open_, [(len(seq), (end.ix, end.iy))] + others)
        return f + self.mh.g2_weight * self.T.lam * (t_stay - t_plan)


def make_mh_mcts(cfg: dict, sc) -> MHMCTS:
    """MH-MCTS for a scenario: low-level settings from `mcts`, the rest from `mh`; planning rng
    seeded from the run, as Dec-MCTS in planners.make_planner."""
    return MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**cfg.get("mcts", {})),
                  MHConfig(**cfg.get("mh", {})), [sc.run_seed, 7])


def split_overrides(over: dict) -> tuple[dict, dict]:
    """Overrides of a planner spec such as "mhmcts:horizon=4,tile=12.5" -> (mcts, mh): keys of
    MHConfig go to the config section `mh`, the rest (low-level Dec-MCTS settings) to `mcts`."""
    mh_keys = set(MHConfig.__dataclass_fields__)
    return ({k: v for k, v in over.items() if k not in mh_keys},
            {k: v for k, v in over.items() if k in mh_keys})
