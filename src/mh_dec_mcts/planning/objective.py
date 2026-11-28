"""Planning objective: expected entropy reduction of the belief by planned looks (tree search).

A robot plans on a working copy of its own belief. A rollout samples a world m ~ belief (per
cell), then every planned look samples observations of m with the sensor rates, applies a
Bayes update and scores the drop in entropy over the footprint. Averaged over rollouts this
is the expected entropy reduction of the plan (one rollout is already within ~1-2 %: a footprint
has thousands of cells).

The planner's update is Bayes only (no LBP, no fusion): neighbourhood smoothing belongs to the
mapper (mapping.belief), never to the planning model.

Cost: planning runs on every `stride`-th cell of the global grid. All footprints use the same
global sub-grid, so overlaps between plans count consistently; drops are scaled by stride**2.

Reward (`reward`): "entropy" (default) scores the drop in binary entropy; "variance" the drop in
sum b(1-b), the expected squared error of the belief against a world drawn from it (the MSE the
missions report, in the planner's own model). Same sampled observations and Bayes update.

Local utility of Dec-MCTS (Best et al. 2019, eq. 1):
    f_r(x_r, x_others) = g(x_r + x_others) - g(empty + x_others)
                       = entropy drop by r's looks after the teammates' looks
which is `sequence_gain(with_others(others, rng), seq, rng)`.
"""

import numpy as np

from mh_dec_mcts.planning.information import binary_entropy

REWARDS = ("entropy", "variance")


def variance(b: np.ndarray) -> np.ndarray:
    """b(1 - b): expected squared error of a cell's belief against m ~ b."""
    return b * (1.0 - b)
from mh_dec_mcts.sensing.lattice import Lattice, Pose
from mh_dec_mcts.sensing.sensor import SensorRates


class State:
    """Working belief b and the sampled world m, both (rows, cols) on the planning sub-grid."""

    __slots__ = ("b", "m")

    def __init__(self, b: np.ndarray, m: np.ndarray):
        self.b, self.m = b, m

    def copy(self) -> "State":
        return State(self.b.copy(), self.m)  # the world never changes within a rollout


class PlanGrid:
    """Strided snapshot of a belief plus cached footprint bounds on that sub-grid."""

    def __init__(self, belief: np.ndarray, lat: Lattice, rates: SensorRates, stride: int = 8,
                 reward: str = "entropy"):
        if reward not in REWARDS:
            raise ValueError(f"unknown reward '{reward}' (known: {', '.join(REWARDS)})")
        self.s, self.lat, self.rates, self.reward = stride, lat, rates, reward
        self._u = binary_entropy if reward == "entropy" else variance
        self.base = np.ascontiguousarray(belief[::stride, ::stride], dtype=float)
        self._fp: dict[Pose, tuple[int, int, int, int]] = {}
        # per level: s0, s1, H(s0), H(s1) - H(s0)   (expected IG = H(z) - H(z|m), linear in p)
        self._k = []
        for k in range(lat.n_alt):
            s0, s1 = rates.at(k)
            h0, h1 = float(binary_entropy(s0)), float(binary_entropy(s1))
            self._k.append((s0, s1, h0, h1 - h0))

    def bounds(self, p: Pose) -> tuple[int, int, int, int]:
        b = self._fp.get(p)
        if b is None:
            s = self.s
            i0, i1, j0, j1 = self.lat.footprint(p)
            b = (-(-i0 // s), -(-i1 // s), -(-j0 // s), -(-j1 // s))  # ceil: same global sub-grid
            self._fp[p] = b
        return b

    def new(self, rng: np.random.Generator) -> State:
        """Start of a rollout: belief = the real belief, world sampled from it."""
        return State(self.base.copy(), rng.random(self.base.shape) < self.base)

    def gain(self, st: State, p: Pose) -> float:
        """Expected drop of the reward (Bayes IG, or of b(1-b)) by one look from p, scaled to full
        resolution. A cheap one-step score for the greedy rollout policy."""
        i0, i1, j0, j1 = self.bounds(p)
        cells = st.b[i0:i1, j0:j1]
        if not cells.size:
            return 0.0
        s0, s1, h0, dh = self._k[p.k]
        if self.reward == "variance":
            return float(np.sum(self.variance_drop(cells, s0, s1))) * self.s * self.s
        pz1 = s0 + (1.0 - s0 - s1) * cells
        return float(np.sum(binary_entropy(pz1)) - h0 * cells.size - dh * np.sum(cells)) * self.s * self.s

    @staticmethod
    def variance_drop(b: np.ndarray, s0: float, s1: float) -> np.ndarray:
        """Per cell: b(1-b) minus its expectation after one observation (P(z=1 | m=0) = s0,
        P(z=0 | m=1) = s1)."""
        pz1 = s0 + (1.0 - s0 - s1) * b
        b1 = (1.0 - s1) * b / np.maximum(pz1, 1e-300)  # posterior after z = 1
        b0 = s1 * b / np.maximum(1.0 - pz1, 1e-300)  # after z = 0
        return variance(b) - pz1 * variance(b1) - (1.0 - pz1) * variance(b0)

    def observe(self, st: State, p: Pose, rng: np.random.Generator) -> float:
        """In place: one look from p (sampled observations of the world, Bayes update).
        Returns the drop of the reward (entropy, or sum b(1-b)) over the footprint, full scale."""
        i0, i1, j0, j1 = self.bounds(p)
        b = st.b[i0:i1, j0:j1]
        if not b.size:
            return 0.0
        h_before = self._u(b).sum()
        s0, s1, _, _ = self._k[p.k]
        m = st.m[i0:i1, j0:j1]
        z = rng.random(b.shape) < np.where(m, 1.0 - s1, s0)  # P(z=1 | m)
        l1 = np.where(z, 1.0 - s1, s1)  # P(z | m=1)
        l0 = np.where(z, s0, 1.0 - s0)  # P(z | m=0)
        b = np.clip(l1 * b / (l1 * b + l0 * (1.0 - b)), 1e-9, 1 - 1e-9)
        st.b[i0:i1, j0:j1] = b
        return float(h_before - self._u(b).sum()) * self.s * self.s

    def with_others(self, others: list[list[Pose]], rng: np.random.Generator) -> State:
        """New rollout state with every teammate's planned looks already taken."""
        st = self.new(rng)
        for seq in others:
            for p in seq:
                self.observe(st, p, rng)
        return st

    def sequence_gain(self, st: State, seq, rng: np.random.Generator, discount: float = 1.0) -> float:
        """Sum of (discounted) entropy drops along seq; st is modified in place."""
        total, w = 0.0, 1.0
        for p in seq:
            total += w * self.observe(st, p, rng)
            w *= discount
        return total

