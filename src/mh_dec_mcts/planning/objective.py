"""Planning objective: expected entropy reduction of the belief by planned looks (tree search).

A robot plans on a working copy of its own belief. A rollout samples a world m ~ belief (per
cell), then every planned look samples observations of m with the sensor rates, applies the
mapping update and scores the drop in entropy over the footprint. Averaged over rollouts this
is the expected entropy reduction of the plan (one rollout is already within ~1-2 %: a footprint
has thousands of cells).

The mapping update matters. With LBP (weights equal / biased / adaptive) one look is worth more
than Bayes alone, and repeated looks are worth almost as much as the first one, because each
cell is pulled by 4 neighbours that see the same smooth ground truth with independent noise.
A Bayes-only planner under-values revisits by ~3x after six looks and plans worse than greedy
IG. So each planning cell carries C = 5 belief copies: the cell and its 4 neighbours, with the
same truth, independent sensor noise and one LBP iteration among them, exactly as in
mapping.belief.lbp_update with the mapper's pairwise weight psi at that altitude (lbp_psi). Mean
cell entropy after 1..6 looks at 21.6 m, adaptive weights:
    real mapper  0.825 0.641 0.457 0.311 0.206 0.139
    5 copies     0.825 0.639 0.460 0.309 0.199 0.123
    Bayes only   0.852 0.733 0.633 0.548 0.475 0.413
weights none: psi=None, one copy, plain Bayes (exact).

Cost: planning runs on every `stride`-th cell of the global grid. All footprints use the same
global sub-grid, so overlaps between plans count consistently; drops are scaled by stride**2.

Local utility of Dec-MCTS (Best et al. 2019, eq. 1):
    f_r(x_r, x_others) = g(x_r + x_others) - g(empty + x_others)
                       = entropy drop by r's looks after the teammates' looks
which is `sequence_gain(with_others(others, rng), seq, rng)`.
"""

import numpy as np

from mh_dec_mcts.sensing.lattice import Lattice, Pose
from mh_dec_mcts.sensing.sensor import SensorRates

N_COPIES = 5  # a cell and its 4 neighbours


def _hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class State:
    """Working belief copies b (C, rows, cols) and the sampled world m (rows, cols)."""

    __slots__ = ("b", "m")

    def __init__(self, b: np.ndarray, m: np.ndarray):
        self.b, self.m = b, m

    def copy(self) -> "State":
        return State(self.b.copy(), self.m)  # the world never changes within a rollout


class PlanGrid:
    """Strided snapshot of a belief plus cached footprint bounds on that sub-grid."""

    def __init__(self, belief: np.ndarray, lat: Lattice, rates: SensorRates, stride: int = 8,
                 psi=None):
        self.s, self.lat, self.rates = stride, lat, rates
        self.base = np.ascontiguousarray(belief[::stride, ::stride], dtype=float)
        lbp = psi is not None and not np.allclose(psi, 0.5)
        self.psi = [float(x) for x in psi] if lbp else None  # per level; None: Bayes only
        self.c = N_COPIES if lbp else 1
        self._fp: dict[Pose, tuple[int, int, int, int]] = {}
        # per level: s0, s1, H(s0), H(s1) - H(s0)   (expected IG = H(z) - H(z|m), linear in p)
        self._k = []
        for k in range(lat.n_alt):
            s0, s1 = rates.at(k)
            h0, h1 = float(_hb(s0)), float(_hb(s1))
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
        """Start of a rollout: belief copies = the real belief, world sampled from it."""
        return State(np.repeat(self.base[None], self.c, axis=0), rng.random(self.base.shape) < self.base)

    def gain(self, st: State, p: Pose) -> float:
        """Expected Bayes IG of looking from p (copy 0), scaled to full resolution. A cheap
        one-step score for the greedy rollout policy."""
        i0, i1, j0, j1 = self.bounds(p)
        cells = st.b[0, i0:i1, j0:j1]
        if not cells.size:
            return 0.0
        s0, s1, h0, dh = self._k[p.k]
        pz1 = s0 + (1.0 - s0 - s1) * cells
        hz = -pz1 * np.log2(pz1) - (1 - pz1) * np.log2(1 - pz1)  # pz1 in [s0, 1 - s1]
        return float(np.sum(hz) - h0 * cells.size - dh * np.sum(cells)) * self.s * self.s

    def observe(self, st: State, p: Pose, rng: np.random.Generator) -> float:
        """In place: one look from p (sampled observations of the world, Bayes, LBP among the
        copies). Returns the entropy drop over the footprint (mean over copies), full scale."""
        i0, i1, j0, j1 = self.bounds(p)
        b = st.b[:, i0:i1, j0:j1]
        if not b[0].size:
            return 0.0
        h_before = _hb(b).sum()
        s0, s1, _, _ = self._k[p.k]
        m = st.m[i0:i1, j0:j1]
        z = rng.random(b.shape) < np.where(m, 1.0 - s1, s0)  # P(z=1 | m)
        l1 = np.where(z, 1.0 - s1, s1)  # P(z | m=1)
        l0 = np.where(z, s0, 1.0 - s0)  # P(z | m=0)
        b = np.clip(l1 * b / (l1 * b + l0 * (1.0 - b)), 1e-9, 1 - 1e-9)
        if self.psi is not None:  # each copy times the messages of the other copies
            psi = self.psi[p.k]
            msg = psi * b + (1.0 - psi) * (1.0 - b)
            l1 = np.log(b) + np.log(msg).sum(0) - np.log(msg)
            l0 = np.log1p(-b) + np.log1p(-msg).sum(0) - np.log1p(-msg)
            b = np.clip(1.0 / (1.0 + np.exp(np.clip(l0 - l1, -50, 50))), 1e-9, 1 - 1e-9)
        st.b[:, i0:i1, j0:j1] = b
        return float(h_before - _hb(b).sum()) / self.c * self.s * self.s

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


def lbp_psi(lat: Lattice, rates: SensorRates, weights: str, field: np.ndarray,
            rng: np.random.Generator, looks: int = 8) -> np.ndarray | None:
    """Mean pairwise weight psi[0, 0] the mapper uses at each altitude level, from `looks`
    synthetic observations at random poses of a calibration field (not an evaluated one).
    None for weights "none" (no LBP)."""
    from mh_dec_mcts.mapping.belief import pairwise_psi
    from mh_dec_mcts.sensing.sensor import observe

    if weights == "none":
        return None
    out = np.empty(lat.n_alt)
    for k in range(lat.n_alt):
        ws = []
        for _ in range(looks):
            b = lat.footprint(Pose(int(rng.integers(lat.nx)), int(rng.integers(lat.ny)), k))
            ws.append(pairwise_psi(weights, observe(field, b, *rates.at(k), rng))[0, 0])
        out[k] = np.mean(ws)
    return out
