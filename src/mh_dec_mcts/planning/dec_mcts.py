"""Dec-MCTS (Best et al., IJRR 2019) for the lattice mapping problem.

Per time step every agent plans an action sequence of length `horizon` on its own belief and
executes only the first action (online replanning, Sec. 4.6; a fresh tree every step since the
belief changed). Planning runs `iterations` rounds of Algorithm 1; in each round every agent:

    1. grows its own tree with `rollouts` D-UCT rollouts (Alg. 2). Each rollout samples one plan
       per known teammate from that teammate's (X_hat, q) and scores the agent's own sequence
       (tree prefix + default policy) with the local utility f_r = entropy drop of its looks
       after the teammates' looks (planning/objective.py: sampled world, Bayes update);
    2. every `resample_every` rounds re-selects X_hat_r = the initial rollout sequences of the
       `n_seq` tree nodes with the best discounted mean, resetting q_r to uniform and beta (Sec. 4.4);
    3. updates q_r by Alg. 3, expectations over `n_samples` sampled teammate plans and worlds;
    4. sends (X_hat_r, q_r) to its neighbours (comms range); each message is lost with prob `drop`.
Then beta <- cool * beta. Action = first move of argmax_x q_r(x) (Alg. 1 line 10).

variant "single" (Kent et al., MH-MCTS paper, Sec. IV-B): each agent sends ONE intent, the
tree's exploitation path (c = 0), refreshed every round; no X_hat / q / Alg. 3; the action is the
first move of that path. Cheaper (no Alg. 3 evaluations); use gamma close to 1 (they use 0.9999).

Teammates never heard from (out of range, or lost messages) are ignored: their plans count as
empty (the paper's "default policy"). With one agent this is plain D-UCT MCTS.

Interface: plan(beliefs, poses, nbrs) -> one action per agent (planners.py), and the single-agent
choose(belief, pose). Build it for a scenario with planners.make_planner("decmcts", cfg, sc).
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from mh_dec_mcts.planning.objective import PlanGrid, State
from mh_dec_mcts.sensing.lattice import Lattice, Pose
from mh_dec_mcts.sensing.sensor import SensorRates

Seq = tuple[Pose, ...]


@dataclass
class DecMCTSConfig:
    horizon: int = 6  # actions per planned sequence
    iterations: int = 30  # rounds n of Alg. 1 per time step
    rollouts: int = 10  # tau: D-UCT rollouts per round (paper Sec. 6: 10)
    n_seq: int = 10  # |X_hat_r| (paper: 10)
    resample_every: int = 10  # rounds between re-selections of X_hat_r (paper: 10)
    n_samples: int = 3  # teammate-plan / world samples for the Alg. 3 expectations
    gamma: float = 0.95  # D-UCT discount, in (1/2, 1)
    cp: float = 0.5  # exploration constant, > 1/sqrt(8)
    alpha: float = 0.1  # Alg. 3 step size (paper's example 0.01 moves q too little in 10 rounds)
    beta0: float = 0.1  # Alg. 3 initial temperature, on utilities scaled to [0, 1]
    cool: float = 0.8  # beta <- cool * beta each round; back to beta0 when X_hat is re-selected
    rollout: str = "random"  # rollout policy: random | greedy
    reward_discount: float = 1.0  # weight of the t-th footprint along a sequence
    stride: int = 1  # plan on every stride-th cell (objective.PlanGrid); 1 = full grid, as greedy.
    #                  > 1 is a speed-up only: the planner sees 1 / stride**2 of the cells
    stride_fine: int | None = None  # coverage-adaptive stride: a UAV plans at `stride` until its
    #                                 coverage reaches stride_cov, then at stride_fine (None: fixed)
    stride_cov: float = 0.9  # coverage = share of cells its belief has seen (b != 0.5: observed or shared)
    drop: float = 0.0  # probability that a message is lost
    share_plans: bool = True  # False: no plan exchange, every agent plans alone (ablation)
    variant: str = "distribution"  # distribution: Best et al. 2019 (X_hat + q, Alg. 3, argmax q)
    #                                single: MH-MCTS paper's simplification, one intent = the tree's
    #                                exploitation path (c = 0), refreshed every round, no Alg. 3


class Node:
    __slots__ = ("pose", "depth", "parent", "children", "untried", "n", "w", "seq")

    def __init__(self, pose: Pose, depth: int, parent, untried: list[str]):
        self.pose, self.depth, self.parent = pose, depth, parent
        self.children: dict[str, Node] = {}
        self.untried = untried  # admissible actions not expanded yet
        self.n = 0.0  # discounted visit count (eq. 4)
        self.w = 0.0  # discounted reward sum (numerator of eq. 6)
        self.seq: Seq = ()  # rollout sequence from its first visit (Sec. 4.4)

    @property
    def mean(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class Tree:
    """One agent's search tree over its own action sequences, with D-UCT statistics."""

    def __init__(self, lat: Lattice, root: Pose, cfg: DecMCTSConfig):
        self.lat, self.cfg = lat, cfg
        self.root = Node(root, 0, None, lat.admissible(root))
        self.nodes: list[Node] = []
        self.f_max = 1e-12  # rewards are scaled by the best seen so far (UCB needs [0, 1])

    def select_expand(self, rng: np.random.Generator) -> Node:
        """Alg. 2 lines 3-4: descend by D-UCT to an expandable node and add one child."""
        node = self.root
        while not node.untried and node.depth < self.cfg.horizon:
            node = self._best_child(node)
        if node.depth >= self.cfg.horizon:
            return node  # full-length leaf: re-evaluate it
        a = node.untried.pop(rng.integers(len(node.untried)))
        pose = self.lat.step(node.pose, a)
        child = Node(pose, node.depth + 1, node, self.lat.admissible(pose))
        node.children[a] = child
        self.nodes.append(child)
        return child

    def _best_child(self, node: Node) -> Node:
        """eq. 2-3, 7: argmax of discounted mean + 2 Cp sqrt(log t_parent / t_j)."""
        t = max(sum(c.n for c in node.children.values()), 1.0)

        def ucb(c: Node) -> float:
            return self.score(c.mean) + 2 * self.cfg.cp * math.sqrt(math.log(t) / max(c.n, 1e-12))
        return max(node.children.values(), key=ucb)

    def score(self, mean: float) -> float:
        """A discounted mean on the [0, 1] scale UCB needs: divided by the best reward seen."""
        return mean / self.f_max

    def prefix(self, node: Node) -> list[Pose]:
        out = []
        while node.parent is not None:
            out.append(node.pose)
            node = node.parent
        return out[::-1]

    def backprop(self, node: Node, f: float) -> None:
        """Alg. 2 line 8. Every visit of a parent discounts all its children (eq. 4, 6)."""
        self.f_max = max(self.f_max, f)
        g = self.cfg.gamma
        while node.parent is not None:
            for c in node.parent.children.values():
                c.n *= g
                c.w *= g
            node.n += 1.0
            node.w += f
            node = node.parent

    def exploit_sequence(self) -> Seq:
        """MH-MCTS paper: descend by the best discounted mean (c = 0) to a leaf; its first-visit
        rollout sequence (path to the leaf + default policy to the horizon)."""
        node = self.root
        while node.children:
            node = max(node.children.values(), key=lambda c: c.mean)
        return node.seq

    def best_sequences(self, k: int) -> list[Seq]:
        """Sec. 4.4: initial rollout sequences of the k nodes with the best discounted mean."""
        ranked = sorted(self.nodes, key=lambda c: c.mean, reverse=True)
        out: list[Seq] = []
        for c in ranked:
            if c.seq and c.seq not in out:
                out.append(c.seq)
            if len(out) == k:
                break
        return out


class DecMCTS:
    """seed: planning randomness (tree expansion, rollouts, sampled plans and observations,
    message loss) comes from its own generator, so the scenario's observation and tie-break
    streams are left exactly as the other planners use them."""

    def __init__(self, lat: Lattice, rates: SensorRates, cfg: DecMCTSConfig | None = None,
                 seed: int | Sequence[int] = 0):
        self.lat, self.rates, self.cfg = lat, rates, cfg or DecMCTSConfig()
        self.rng = np.random.default_rng(seed)
        self.comm_rng = np.random.default_rng(self.rng.integers(2**62))  # message loss only
        self.last_q: list[tuple[list[Seq], np.ndarray]] = []  # per agent, for inspection
        self.last_trees: list[Tree] = []
        self.last_strides: list[int] = []  # per agent, the stride of the last plan()

    # ---------- single agent ----------
    def choose(self, belief, pose, visited=None, rng=None) -> str:
        return self.plan([belief], [pose], [[]])[0]

    # ---------- team (planners.py interface) ----------
    def plan(self, beliefs, poses, nbrs) -> list[str]:
        """beliefs, poses: per agent; nbrs[r]: who hears agent r (comms.network.neighbours)."""
        cfg, n = self.cfg, len(poses)
        if cfg.variant not in ("distribution", "single"):
            raise ValueError(f"unknown Dec-MCTS variant '{cfg.variant}'")
        if not cfg.share_plans:
            nbrs = [[] for _ in range(n)]
        grids = [PlanGrid(b, self.lat, self.rates, self.stride_for(b)) for b in beliefs]
        self.last_strides = [g.s for g in grids]
        self._start_plan(grids)
        trees = [self._new_tree(p) for p in poses]
        own: list[list[Seq]] = [[] for _ in range(n)]  # X_hat_r
        q: list[np.ndarray] = [np.zeros(0) for _ in range(n)]
        known: list[dict[int, tuple[list[Seq], np.ndarray]]] = [{} for _ in range(n)]  # r's copy of others
        caches: list[dict] = [{} for _ in range(n)]  # per teammate sample: state after its looks
        beta = cfg.beta0
        for it in range(cfg.iterations):
            for r in range(n):
                ctx = self._rollout_context(r, known[r])
                for _ in range(cfg.rollouts):
                    self._rollout(trees[r], grids[r], known[r], caches[r], ctx)
                if cfg.variant == "single":
                    own[r], q[r] = [trees[r].exploit_sequence()], np.ones(1)
                else:
                    if it % cfg.resample_every == 0:
                        own[r] = trees[r].best_sequences(cfg.n_seq)
                        q[r] = np.full(len(own[r]), 1.0 / len(own[r]))
                    q[r] = self._update_q(grids[r], own[r], q[r], known[r], caches[r], beta, trees[r].f_max)
                self._round_done(r, trees[r], known[r])
            for r in range(n):  # communication: the message replaces the receiver's copy
                for j in nbrs[r]:
                    if cfg.drop == 0.0 or self.comm_rng.random() >= cfg.drop:
                        if r in known[j] and known[j][r][0] != own[r]:
                            caches[j].clear()  # cached entries used r's previous X_hat
                        known[j][r] = (own[r], q[r].copy())
                        self._delivered(r, j)
            beta = cfg.beta0 if (it + 1) % cfg.resample_every == 0 else beta * cfg.cool
        self.last_q, self.last_trees = list(zip(own, q)), trees
        return [self._first_action(poses[r], own[r][int(np.argmax(q[r]))]) for r in range(n)]

    def stride_for(self, belief: np.ndarray) -> int:
        """The planning stride for one UAV this step: `stride`, or with stride_fine set, stride_fine
        once the UAV's own coverage (cells its belief has seen) reaches stride_cov. Unseen cells
        hold the prior 0.5 exactly (Bayes and LBP only touch the footprint)."""
        cfg = self.cfg
        if cfg.stride_fine is None:
            return cfg.stride
        return cfg.stride_fine if np.mean(belief != 0.5) >= cfg.stride_cov else cfg.stride

    # ---------- hooks (planners built on Dec-MCTS override these; see mh_mcts.MHMCTS) ----------
    def _start_plan(self, grids: list[PlanGrid]) -> None:
        """Called once per plan() with every agent's planning grid, before any tree is grown."""

    def _new_tree(self, root: Pose) -> Tree:
        return Tree(self.lat, root, self.cfg)

    def _rollout_context(self, r: int, known: dict) -> object:
        """Whatever agent r's rollouts of this round need besides the tree, passed on to
        _tree_reward (Dec-MCTS: nothing)."""

    def _round_done(self, r: int, tree: Tree, known: dict) -> None:
        """Called after agent r's rollouts of a round and its new intent (Dec-MCTS: nothing)."""

    def _delivered(self, sender: int, receiver: int) -> None:
        """Called when sender's message of this round reaches receiver (Dec-MCTS: nothing)."""

    def _tree_reward(self, tree: Tree, grid: PlanGrid, g: State, seq: list[Pose], ctx) -> float:
        """Utility of the tree part `seq` of a rollout; takes its looks on g in place."""
        return grid.sequence_gain(g, seq, self.rng, self.cfg.reward_discount)

    # ---------- pieces ----------
    def _sample_others(self, known) -> tuple:
        """One plan per known teammate drawn from its q: ((teammate, index), ...)."""
        return tuple((j, int(self.rng.choice(len(seqs), p=qj))) for j, (seqs, qj) in sorted(known.items()))

    def _grid_after(self, grid: PlanGrid, known, cache: dict, sample: tuple) -> State:
        """Rollout state after the sampled teammate plans. With teammates, one sampled world
        and realisation per teammate sample is cached (their looks are the expensive part);
        without, every rollout gets a fresh world."""
        if not sample:
            return grid.new(self.rng)
        st = cache.get(("grid", sample))
        if st is None:
            st = grid.with_others([known[j][0][i] for j, i in sample], self.rng)
            cache[("grid", sample)] = st
        return st.copy()

    def _rollout(self, tree: Tree, grid: PlanGrid, known, cache, ctx=None) -> None:
        """Alg. 2 lines 3-8: tree prefix, then the default policy up to the horizon; the
        utility is accumulated on the grid after the sampled teammate plans."""
        node = tree.select_expand(self.rng)
        g = self._grid_after(grid, known, cache, self._sample_others(known))
        seq = tree.prefix(node)
        f = self._tree_reward(tree, grid, g, seq, ctx)
        w = self.cfg.reward_discount ** len(seq)
        pose = node.pose
        for _ in range(self.cfg.horizon - node.depth):  # default policy
            nxt = [q for a in self.lat.admissible(pose) if (q := self.lat.move(pose, a)) is not None]
            if self.cfg.rollout == "greedy":  # best expected (Bayes) IG of the next look
                pose = nxt[int(np.argmax([grid.gain(g, q) for q in nxt]))]
            else:
                pose = nxt[self.rng.integers(len(nxt))]
            seq.append(pose)
            f += w * grid.observe(g, pose, self.rng)
            w *= self.cfg.reward_discount
        if not node.seq:
            node.seq = tuple(seq)
        tree.backprop(node, f)

    def _update_q(self, grid, seqs, q, known, cache, beta, f_max) -> np.ndarray:
        """Alg. 3 with the expectations over `n_samples` sampled teammate plans (and worlds);
        every x is scored on the same samples (common random numbers). Utilities / f_max."""
        states = [self._grid_after(grid, known, cache, self._sample_others(known))
                  for _ in range(self.cfg.n_samples)]
        f = np.array([[grid.sequence_gain(st.copy(), x, self.rng, self.cfg.reward_discount)
                       for st in states] for x in seqs]) / f_max
        e_cond = f.mean(axis=1)  # E[f_r | x_r]
        e = float(q @ e_cond)  # E[f_r]
        h = -float(np.sum(q * np.log(q)))
        q = q - self.cfg.alpha * q * ((e - e_cond) / beta + h + np.log(q))
        q = np.clip(q, 1e-12, None)
        return q / q.sum()

    def _first_action(self, pose: Pose, seq: Seq) -> str:
        for a in self.lat.admissible(pose):
            if self.lat.move(pose, a) == seq[0]:
                return a
        raise RuntimeError(f"sequence {seq[:2]} does not start next to {pose}")

