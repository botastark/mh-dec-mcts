"""For one UAV, sim.multi.run_multi reproduces sim.single.run_single (the loop behind Pierdicca's
Figs. 3 and 4) exactly: poses, MSE, entropy, altitude, and the legacy coverage rebuilt from poses.
Same experiment structure as the old run_single.py: per planner, one `streams` (compat) and the
runs in order."""

import numpy as np
import pytest

from mh_dec_mcts.experiment import paper_coverage
from mh_dec_mcts.planning.planners import make_planner
from mh_dec_mcts.scenario import load_config, make_scenario, make_streams, mapping_weights
from mh_dec_mcts.sim.multi import run_multi
from mh_dec_mcts.sim.single import run_single

RUNS, STEPS = 2, 30


@pytest.mark.parametrize("config", ["configs/paper.yaml", "configs/pierdicca_fig3.yaml"])
@pytest.mark.parametrize("planner", ["ig", "random", "sweep"])
def test_run_multi_equals_run_single_for_one_uav(config, planner):
    cfg = load_config(config, agents={"n": 1})
    w = mapping_weights(cfg)
    s_streams, m_streams = make_streams(cfg), make_streams(cfg)
    for i in range(RUNS):
        sc = make_scenario(cfg, run=i, streams=s_streams)
        a = run_single(sc, make_planner(planner, cfg, sc), STEPS, w)
        sc = make_scenario(cfg, run=i, streams=m_streams)
        b = run_multi(sc, make_planner(planner, cfg, sc, 5), STEPS, w, "none", 5)
        assert a.poses == b.poses[0]
        assert a.mse == [m[0] for m in b.mse]
        assert a.entropy == [h[0] for h in b.entropy]
        assert a.altitude == [h[0] for h in b.altitude]
        assert a.coverage == paper_coverage(sc.lattice, sc.grid.shape, b.poses, STEPS)
        np.testing.assert_array_equal(a.belief, b.beliefs[0])
