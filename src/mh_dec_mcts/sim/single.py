"""Single-agent run (legacy simulator loop for N = 1).

Per step, as in the legacy simulator:  metrics -> plan -> move -> observe -> update.
So step 0 records the prior, and no observation is taken at the start pose.
"""

from dataclasses import dataclass, field

import numpy as np

from mh_dec_mcts.mapping.belief import update
from mh_dec_mcts.scenario import Scenario
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.sim.metrics import coverage, mse, total_entropy

METRICS = ("entropy", "mse", "coverage", "altitude")


@dataclass
class RunResult:
    poses: list[Pose]  # start + one per step
    actions: list[str]
    entropy: list[float] = field(default_factory=list)
    mse: list[float] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    altitude: list[float] = field(default_factory=list)
    belief: np.ndarray | None = None  # final belief


def run_single(sc: Scenario, planner, n_steps: int, weights: str = "equal") -> RunResult:
    """planner: planning.planners.make_planner(name, cfg, sc), fresh for this run."""
    lat = sc.lattice
    belief = np.full(sc.grid.shape, 0.5)
    seen = np.zeros(sc.grid.shape, dtype=bool)
    pose = sc.starts[0]
    res = RunResult(poses=[pose], actions=[])
    for _ in range(n_steps):
        # 1. metrics (legacy counts the current footprint as covered before planning)
        i0, i1, j0, j1 = lat.footprint(pose)
        seen[i0:i1, j0:j1] = True
        res.entropy.append(total_entropy(belief))
        res.mse.append(mse(belief, sc.truth))
        res.coverage.append(coverage(seen))
        res.altitude.append(lat.xyz(pose)[2])
        # 2. plan, 3. move
        action = planner.plan([belief], [pose], [[]])[0]
        pose = lat.step(pose, action)
        res.actions.append(action)
        res.poses.append(pose)
        # 4. observe, 5. update
        update(belief, lat.footprint(pose), sc.observe(pose), *sc.rates.at(pose.k), weights)
    res.belief = belief
    return res
