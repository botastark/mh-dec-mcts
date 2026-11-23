"""Multi-agent run (paper Sec. III-D/E, Fig. 5), following Pierdicca's simulate.py.

Per step:
    1. metrics: per-agent entropy / MSE / altitude, misalignment between agents
    2. plan: any planner of planning/planners.py (make_planner), given the neighbours at the
       current poses (Dec-MCTS exchanges its plans with them)
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
from mh_dec_mcts.scenario import Scenario
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.sim.metrics import misalignment, mse, total_entropy


@dataclass
class MultiResult:
    poses: list[list[Pose]]  # per agent: start + one per step
    entropy: list[list[float]] = field(default_factory=list)  # [step][agent]
    mse: list[list[float]] = field(default_factory=list)
    altitude: list[list[float]] = field(default_factory=list)
    misalignment: list[float] = field(default_factory=list)  # [step]
    beliefs: list[np.ndarray] | None = None  # final, per agent


def run_multi(sc: Scenario, planner, n_steps: int, weights: str = "adaptive", sharing: str = "none",
              range_: int | None = None) -> MultiResult:
    """planner: planning.planners.make_planner(name, cfg, sc, range_), fresh for this run."""
    lat, n = sc.lattice, len(sc.starts)
    beliefs = [np.full(sc.grid.shape, 0.5) for _ in range(n)]
    news = News(sharing, n, sc.grid.shape)
    poses = list(sc.starts)
    res = MultiResult(poses=[[p] for p in poses])
    for _ in range(n_steps):
        # 1. metrics
        res.entropy.append([total_entropy(b) for b in beliefs])
        res.mse.append([mse(b, sc.truth) for b in beliefs])
        res.altitude.append([lat.xyz(p)[2] for p in poses])
        res.misalignment.append(misalignment(beliefs))
        # 2. plan, 3. move
        actions = planner.plan(beliefs, poses, neighbours(poses, range_))
        poses = [lat.step(p, a) for p, a in zip(poses, actions)]
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
        for a, psi in enumerate(psis):
            if psi is not None:  # weights "none": Bayes only
                lbp_update(beliefs[a], bounds[a], psi)
        # 6. news
        news.step(beliefs, bounds, zs, rates, psis, nbrs)
    res.beliefs = beliefs
    return res
