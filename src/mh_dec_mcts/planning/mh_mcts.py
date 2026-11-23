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
        saved into bits. With no open region left, g2 = 0 and the planner is plain Dec-MCTS
        with the short horizon; w = 0 gives Dec-MCTS (single variant) as well.
Step 5: the high-level planner (HLP, paper Sec. V-VI). Every UAV also grows a RegionTree: an
        action is the next region, a path an order of the open regions, terminal when none is
        left; below the tree the order is completed nearest-first (`hl_rollout: nearest`) or at
        random (the paper's default policy, `random`). Its reward is
            w * lam * (T[own order = nearest-first] - T[own order = this order])
        with every known low-level plan fixed (one sampled realisation per round). Each round a
        UAV grows its low-level tree, then its HL tree (`hl_rollouts` expansions), and sends both
        intents (low-level sequence and order) in one message. In g2, every UAV follows its order
        (MissionTime); the low-level reward thus pulls towards the first open region of the
        UAV's own order. The HL tree is kept across time steps and rebuilt only when the set of
        open regions changes incompatibly (paper: hysteresis, gamma 0.9999): regions that close
        are dropped from the search; it is rebuilt when a region reopens or fewer than
        `hl_rebuild` of its regions are left. hl_rollouts = 0: no HLP, every UAV goes
        nearest-first (step 3).
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
Order = tuple[int, ...]  # a high-level plan: regions in the order a UAV intends to finish them


@dataclass
class MHConfig:
    tile: float = 25.0  # region side, metres (100 x 50 m field -> 4 x 2 regions)
    done_entropy: float = 0.3  # a region is finished below this mean cell entropy (bits; prior = 1)
    rate: float | str = "auto"  # lam, entropy drop per step in bits ("auto": look_rate at level 0)
    g2_weight: float = 1.0  # w in f = g1 + w lam (T_stay - T_plan); 0 = Dec-MCTS
    hl_rollouts: int = 10  # high-level expansions per UAV per round; 0 = no HLP (nearest-first)
    hl_gamma: float = 0.9999  # D-UCT discount of the HL tree (paper: 0.9999; it lives many steps)
    hl_cp: float = 0.5  # exploration constant of the HL tree
    hl_rollout: str = "nearest"  # HL default policy: nearest (open region nearest to the last) | random
    hl_rebuild: float = 0.5  # rebuild the HL tree once fewer than this share of its regions are open


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
        self._box = np.array(self.box)  # (n, 4)
        self.from_center = self.dist_all(np.array(self.center))  # (n, n): centre of r -> region j

    def entropy(self, b0: np.ndarray) -> np.ndarray:
        """Entropy (bits, full resolution) of each region of a sub-grid belief."""
        return np.bincount(self.label, weights=binary_entropy(b0).ravel(), minlength=self.n) * self.scale

    def mean_entropy(self, b0: np.ndarray) -> np.ndarray:
        """Mean cell entropy (bits) of each region; 1 = unknown, 0 = certain."""
        return self.entropy(b0) / (self.cells * self.scale)

    def dist_all(self, xy) -> np.ndarray:
        """Waypoint steps from waypoint xy = (ix, iy) (or an array of them, shape (..., 2)) to
        every region: shape (..., n)."""
        xy = np.asarray(xy)
        x, y = xy[..., 0:1], xy[..., 1:2]
        a, b, c, d = self._box.T
        return np.maximum(0, np.maximum(a - x, x - b)) + np.maximum(0, np.maximum(c - y, y - d))

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
    earliest time takes its next region and spends
        travel  waypoint steps to the region (Regions.dist), plus
        work    the region's entropy above the closing level, divided by `lam` (bits/step),
    on it, and continues from the region's centre. A UAV's next region is the first one of its
    order (high-level plan) that is still open; without an order, or once it is used up, the
    nearest open region (lowest id on ties). The estimate is the mean of the UAVs' finishing
    times, as in the paper (a UAV that gets no region finishes at its start time).

    Cost: one numpy pass over the regions per region assigned. For a single UAV the work does not
    depend on the order, so T = t0 + travel of its tour + total work, and nearest-first tours are
    cached per (start waypoint, open regions, order): most rollouts of a planning step repeat one.
    """

    def __init__(self, reg: Regions, lam: float, done: float):
        self.reg, self.lam, self.done = reg, lam, done
        self._tours: dict = {}  # (start waypoint, open regions) -> nearest-first travel

    def open(self, b0: np.ndarray) -> frozenset[int]:
        """Regions not finished yet in the sub-grid belief b0."""
        return frozenset(int(r) for r in np.flatnonzero(self.reg.mean_entropy(b0) > self.done))

    def work(self, E: np.ndarray, r: int) -> float:
        """Steps to bring region r (entropies E, bits) down to the closing level."""
        return max(0.0, E[r] - self.done * self.reg.cells[r] * self.reg.scale) / self.lam

    def works(self, E: np.ndarray) -> np.ndarray:
        """work() of every region."""
        return np.maximum(0.0, E - self.done * self.reg.cells * self.reg.scale) / self.lam

    def __call__(self, E: np.ndarray, open_: frozenset[int], robots: list[Robot],
                 orders: list[Order | None] | None = None) -> float:
        """robots: [(t0, (ix, iy))], each UAV's time and waypoint at the end of its plan;
        orders: per UAV, its order of regions (high-level plan) or None for nearest-first."""
        orders = orders or [None] * len(robots)
        if not open_:
            return float(np.mean([t0 for t0, _ in robots]))
        w = self.works(E)
        if len(robots) == 1:
            (t0, xy), order = robots[0], orders[0]
            return t0 + self._travel(xy, open_, order) + float(w[sorted(open_)].sum())
        is_open = np.zeros(self.reg.n, bool)
        is_open[list(open_)] = True
        t = [float(t0) for t0, _ in robots]
        d = [self.reg.dist_all(xy) for _, xy in robots]  # per UAV: steps to every region
        nxt_in_order = [0] * len(robots)
        for _ in range(len(open_)):
            i = int(np.argmin(t))
            r = self._next(i, orders[i], nxt_in_order, is_open, d[i])
            t[i] += d[i][r] + w[r]
            d[i] = self.reg.from_center[r]
            is_open[r] = False
        return float(np.mean(t))

    def _next(self, i: int, order, pointer: list[int], is_open: np.ndarray, d: np.ndarray) -> int:
        """UAV i's next region: the first still-open one of its order, else the nearest open."""
        if order:
            while pointer[i] < len(order) and not is_open[order[pointer[i]]]:
                pointer[i] += 1
            if pointer[i] < len(order):
                return order[pointer[i]]
        return int(np.argmin(np.where(is_open, d, np.iinfo(np.int64).max)))

    def _travel(self, xy: tuple[int, int], open_: frozenset[int], order) -> int:
        """Travel of one UAV from xy through all open regions (its order, then nearest-first),
        cached per (start, open regions, order)."""
        key = (xy, open_, tuple(order or ()))
        if key not in self._tours:
            if len(self._tours) > 20000:
                self._tours.clear()
            self._tours[key] = self._tour_travel(xy, open_, order)
        return self._tours[key]

    def _tour_travel(self, xy, open_: frozenset[int], order) -> int:
        is_open = np.zeros(self.reg.n, bool)
        is_open[list(open_)] = True
        d, total, pointer = self.reg.dist_all(xy), 0, [0]
        for _ in range(len(open_)):
            r = self._next(0, order, pointer, is_open, d)
            total += int(d[r])
            d = self.reg.from_center[r]
            is_open[r] = False
        return total


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


class _HLConfig:
    """What Tree reads from its config, for the HL tree."""

    def __init__(self, horizon: int, gamma: float, cp: float):
        self.horizon, self.gamma, self.cp = horizon, gamma, cp


class RegionTree(RangeTree):
    """HL tree of one UAV over the orders of the regions `key` that were open when it was built.
    A node's `pose` is a region id, its path from the root the start of an order; `seq` of a node
    is the full order of its first visit (path + default policy), as for the low-level tree.

    Regions close while the tree lives (`close`): their nodes are kept but no longer selected,
    expanded or reported, so the search over the remaining regions continues (paper: the HL tree
    is only invalidated by an incompatible change of the problem)."""

    def __init__(self, regions: frozenset[int], gamma: float, cp: float):
        self.lat, self.cfg, self.key = None, _HLConfig(len(regions), gamma, cp), regions
        self.open = regions  # still open, a subset of key
        self.root = Node(None, 0, None, sorted(regions))
        self.nodes: list[Node] = []
        self.f_min, self.f_max = math.inf, -math.inf

    def close(self, open_: frozenset[int]) -> None:
        """The open regions are now open_ (a subset of key)."""
        self.open = open_

    def _live(self, node: Node) -> list[Node]:
        return [c for c in node.children.values() if c.pose in self.open]

    def select_expand(self, rng: np.random.Generator) -> Node:
        node = self.root
        while True:
            node.untried = [a for a in node.untried if a in self.open]
            live = self._live(node)
            if node.untried or not live:
                break
            node = max(live, key=lambda c: self._ucb(node, c, live))
        if not node.untried:
            return node  # a complete order of the open regions: re-evaluate it
        a = node.untried.pop(rng.integers(len(node.untried)))
        used = set(self.prefix(node)) | {a}
        child = Node(a, node.depth + 1, node, [x for x in sorted(self.open) if x not in used])
        node.children[a] = child
        self.nodes.append(child)
        return child

    def _ucb(self, node: Node, c: Node, live: list[Node]) -> float:
        t = max(sum(x.n for x in live), 1.0)
        return self.score(c.mean) + 2 * self.cfg.cp * math.sqrt(math.log(t) / max(c.n, 1e-12))

    def exploit_sequence(self) -> Order:
        """Descend by the best discounted mean over open regions; that node's first-visit order,
        without the regions closed since."""
        node = self.root
        while live := self._live(node):
            node = max(live, key=lambda c: c.mean)
        return tuple(x for x in node.seq if x in self.open)

    def complete(self, node: Node, rng: np.random.Generator, reg: Regions | None,
                 start: tuple[int, int]) -> Order:
        """The node's path, then the other open regions: nearest-first from the last one (from
        `start` if the path is empty) when reg is given, else in random order."""
        order = self.prefix(node)
        rest = [x for x in sorted(self.open) if x not in set(order)]
        if reg is None:
            return tuple(order + [rest[i] for i in rng.permutation(len(rest))])
        d = reg.from_center[order[-1]] if order else reg.dist_all(start)
        while rest:
            k = int(np.argmin(d[rest]))  # lowest id on ties: rest is sorted
            order.append(rest.pop(k))
            d = reg.from_center[order[-1]]
        return tuple(order)


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
        self.hl_trees: dict[int, RegionTree] = {}  # per UAV, kept across plan() calls
        self.last_hl: list[Order | None] = []  # per UAV, its current order (HL intent)
        self._hl_known: list[dict[int, Order | None]] = []  # per UAV, teammates' orders received
        self._grids: list[PlanGrid] = []

    def _start_plan(self, grids) -> None:
        self._grids = grids
        self.last_open = [self.T.open(g.base) for g in grids]  # each UAV judges with its own belief
        for r, op in enumerate(self.last_open):  # HL trees persist while regions only close
            tree = self.hl_trees.get(r)
            if not self.mh.hl_rollouts or not op:
                continue
            if tree is None or not op <= tree.key or len(op) < self.mh.hl_rebuild * len(tree.key):
                self.hl_trees[r] = RegionTree(op, self.mh.hl_gamma, self.mh.hl_cp)
            else:
                tree.close(op)
        self.last_hl = [self._hl_intent(r) for r in range(len(grids))]
        self._hl_known = [{} for _ in grids]

    def _hl_intent(self, r: int) -> Order | None:
        tree = self.hl_trees.get(r)
        if not self.mh.hl_rollouts or not self.last_open[r] or tree is None or tree.open != self.last_open[r]:
            return None
        return tree.exploit_sequence() or None

    def _new_tree(self, root: Pose) -> Tree:
        return (RangeTree if self.mh.g2_weight else Tree)(self.lat, root, self.cfg)  # w = 0: Dec-MCTS

    def _teammates(self, r: int, known: dict) -> tuple[list[Robot], list[Order | None]]:
        """Agent r's teammates in g2: their last shared plans as (length, end waypoint), and the
        orders that came with them."""
        mates = [(j, s[0]) for j, (s, _) in sorted(known.items()) if s[0]]
        return ([(len(seq), (seq[-1].ix, seq[-1].iy)) for _, seq in mates],
                [self._hl_known[r].get(j) for j, _ in mates])

    def _rollout_context(self, r: int, known: dict):
        """Agent r's open regions, its order, and its teammates (plans and orders)."""
        return (self.last_open[r], self.last_hl[r], *self._teammates(r, known))

    def _tree_reward(self, tree, grid, g, seq, ctx) -> float:
        """g1 of the tree part plus w lam (T[stay] - T[tree part]); the rollout part gets g1 only."""
        open_, mine, others, theirs = ctx
        if not open_ or self.mh.g2_weight == 0:
            return super()._tree_reward(tree, grid, g, seq, ctx)
        pose = tree.root.pose
        e_stay = self.reg.entropy(g.b)  # after the teammates' looks, before own
        f = super()._tree_reward(tree, grid, g, seq, ctx)
        e_plan = self.reg.entropy(g.b) if seq else e_stay
        end = seq[-1] if seq else pose
        orders = [mine] + theirs
        t_stay = self.T(e_stay, open_, [(0, (pose.ix, pose.iy))] + others, orders)
        t_plan = self.T(e_plan, open_, [(len(seq), (end.ix, end.iy))] + others, orders)
        return f + self.mh.g2_weight * self.T.lam * (t_stay - t_plan)

    def _round_done(self, r: int, tree: Tree, known: dict) -> None:
        """Agent r's HL rollouts of this round, against its new low-level intent and the known
        teammates' plans and orders (one sampled realisation of all their looks)."""
        hl, open_ = self.hl_trees.get(r), self.last_open[r]
        if not self.mh.hl_rollouts or not self.mh.g2_weight or hl is None or hl.open != open_:
            return
        ll = tree.exploit_sequence()
        pose = tree.root.pose
        me = (len(ll), (ll[-1].ix, ll[-1].iy)) if ll else (0, (pose.ix, pose.iy))
        others, theirs = self._teammates(r, known)
        st = self._grids[r].with_others([ll] + [s[0] for s, _ in known.values()], self.rng)
        E = self.reg.entropy(st.b)
        t_null = self.T(E, open_, [me] + others, [None] + theirs)
        reg = self.reg if self.mh.hl_rollout == "nearest" else None
        for _ in range(self.mh.hl_rollouts):
            node = hl.select_expand(self.rng)
            order = hl.complete(node, self.rng, reg, me[1])
            if not node.seq:
                node.seq = order
            t = self.T(E, open_, [me] + others, [order] + theirs)
            hl.backprop(node, self.mh.g2_weight * self.T.lam * (t_null - t))
        self.last_hl[r] = hl.exploit_sequence() or None

    def _delivered(self, sender: int, receiver: int) -> None:
        self._hl_known[receiver][sender] = self.last_hl[sender]  # the order travels with the plan


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
