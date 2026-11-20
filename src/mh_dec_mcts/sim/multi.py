"""Multi-agent run (paper Sec. III-D/E, Fig. 5), following Pierdicca's simulate.py.

Per step:
    1. metrics: per-agent entropy / MSE / altitude, misalignment between agents
    2. plan: IG (each agent greedy on its own belief), IGd (IG discounted by overlap, agents
       deciding one after another in a random order) or Dec-MCTS (planning/dec_mcts.py; plans
       are exchanged with the neighbours at the current poses)
    3. move all agents
    4. each agent observes; neighbours are found at the new positions
    5. each agent updates its own belief (Bayes + LBP with its own pairwise weights)
    6. news sharing and fusion (none / BS / BM)
"""

from dataclasses import dataclass, field

import numpy as np

from mh_dec_mcts.comms.network import neighbours
from mh_dec_mcts.comms.news import News
from mh_dec_mcts.mapping.belief import bayes_update, lbp_update, pairwise_psi
from mh_dec_mcts.planning.dec_mcts import DecMCTS
from mh_dec_mcts.planning.information import entropy
from mh_dec_mcts.planning.planners import GreedyIG, pick_best
from mh_dec_mcts.scenario import Scenario
from mh_dec_mcts.sensing.lattice import Pose


@dataclass
class MultiResult:
    poses: list[list[Pose]]  # per agent: start + one per step
    entropy: list[list[float]] = field(default_factory=list)  # [step][agent]
    mse: list[list[float]] = field(default_factory=list)
    altitude: list[list[float]] = field(default_factory=list)
    misalignment: list[float] = field(default_factory=list)  # [step]
    beliefs: list[np.ndarray] | None = None  # final, per agent


def iou(a, b) -> float:
    """Intersection over union of two footprints (i0, i1, j0, j1), half-open."""
    di = min(a[1], b[1]) - max(a[0], b[0])
    dj = min(a[3], b[3]) - max(a[2], b[2])
    if di < 0 or dj < 0:
        return 0.0
    inter = di * dj
    return inter / float((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter)


def choose_actions(sc: Scenario, planner: str, beliefs, poses, range_, tie_eps: float) -> list[str]:
    """IG: every agent greedy on its own belief. IGd: after the IG pass, agents re-decide in a
    random order; each candidate's IG is multiplied by (1 - IoU) with every neighbour's footprint
    at the neighbour's predicted pose (already-decided agents at their new pose)."""
    g = GreedyIG(sc.lattice, sc.rates, tie_eps)
    scores = [g.scores(beliefs[a], poses[a]) for a in range(len(poses))]
    actions = [pick_best(scores[a], sc.plan_rngs[a], tie_eps) for a in range(len(poses))]
    if planner == "ig" or len(poses) == 1:
        return actions
    if planner != "igd":
        raise ValueError(f"unknown multi-agent planner '{planner}'")
    lat = sc.lattice
    predicted = list(poses)
    actions = ["hover"] * len(poses)
    for a in sc.order_rng.permutation(len(poses)):
        for nb in neighbours(predicted, range_)[a]:
            nb_fp = lat.footprint(predicted[nb])
            for act in scores[a]:
                scores[a][act] *= 1.0 - iou(nb_fp, lat.footprint(lat.move(poses[a], act)))
        actions[a] = pick_best(scores[a], sc.plan_rngs[a], tie_eps)
        predicted[a] = lat.move(poses[a], actions[a])
    return actions


def run_multi(sc: Scenario, planner: str, n_steps: int, weights: str = "adaptive", sharing: str = "none",
              range_: int | None = None, tie_eps: float = 1e-4, dec: DecMCTS | None = None) -> MultiResult:
    """planner "decmcts" needs dec = planning.dec_mcts.make_dec_mcts(cfg, sc)."""
    lat, n = sc.lattice, len(sc.starts)
    if (planner == "decmcts") != (dec is not None):
        raise ValueError("planner 'decmcts' needs dec=make_dec_mcts(cfg, sc), and only it")
    beliefs = [np.full(sc.grid.shape, 0.5) for _ in range(n)]
    news = News(sharing, n, sc.grid.shape)
    poses = list(sc.starts)
    res = MultiResult(poses=[[p] for p in poses])
    for _ in range(n_steps):
        # 1. metrics
        res.entropy.append([float(entropy(b).sum()) for b in beliefs])
        res.mse.append([float(np.mean((sc.truth - b) ** 2)) for b in beliefs])
        res.altitude.append([lat.xyz(p)[2] for p in poses])
        res.misalignment.append(float(np.var(np.stack(beliefs, axis=2), axis=2).mean()))
        # 2. plan, 3. move
        if dec is not None:
            actions = dec.plan(beliefs, poses, neighbours(poses, range_))
        else:
            actions = choose_actions(sc, planner, beliefs, poses, range_, tie_eps)
        poses = [lat.move(p, a) for p, a in zip(poses, actions)]
        if any(p is None for p in poses):
            raise RuntimeError(f"inadmissible action in {actions}")
        for a in range(n):
            res.poses[a].append(poses[a])
        # 4. observe
        bounds = [lat.footprint(p) for p in poses]
        zs = [sc.observe(p, a) for a, p in enumerate(poses)]
        rates = [sc.rates.at(p.k) for p in poses]
        psis = [pairwise_psi(weights, z) if weights != "none" else None for z in zs]
        nbrs = neighbours(poses, range_)
        # 5. own beliefs (Bayes for all agents, then LBP, as in his update_map_beliefs)
        for a in range(n):
            bayes_update(beliefs[a], bounds[a], zs[a], *rates[a])
        if weights != "none":
            for a in range(n):
                lbp_update(beliefs[a], bounds[a], psis[a])
        # 6. news
        news.step(beliefs, bounds, zs, rates, psis, nbrs)
    res.beliefs = beliefs
    return res
