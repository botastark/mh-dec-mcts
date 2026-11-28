"""Options for the Tier 2-3 experiments, each off by default: the b(1-b) reward, root-value logging,
planner seeds, the phase switch to greedy, MH with a greedy low level (mhgreedy, budget-aware
mission time) and the common-depot start."""

from typing import Any

import numpy as np
import pytest

from mh_dec_mcts.experiment import one_run, run_record
from mh_dec_mcts.planning.dec_mcts import DecMCTS, DecMCTSConfig
from mh_dec_mcts.planning.mh_mcts import MHConfig, MHGreedy, MHMCTS, MissionTime, Regions
from mh_dec_mcts.planning.objective import PlanGrid, variance
from mh_dec_mcts.planning.planners import PhaseSwitch, make_planner
from mh_dec_mcts.scenario import make_scenario
from mh_dec_mcts.sensing.lattice import Pose

CFG: dict[str, Any] = {"name": "t", "grid": {"size_x": 50, "size_y": 25, "cell": 0.125}, "lattice": {"xy_step": 3.125},
                       "field": {"type": "gaussian", "r": 5, "ref": 50.0}, "agents": {"n": 2, "start": "legacy"},
                       "run": {"weights": "adaptive"},
                       "mcts": {"variant": "single", "horizon": 2, "iterations": 3, "rollouts": 4, "stride": 8,
                                "rollout": "greedy"},
                       "mh": {"tile": 12.5}}
IGD = {"sharing": "BM", "range": 5, "greedy": "igd"}
IG = {"sharing": "none", "range": 5, "greedy": "ig"}


def cfg_n(n):
    return {**CFG, "agents": {**CFG["agents"], "n": n}}


# ------------------------------------------------------------------ b(1-b) reward
def test_variance_reward_scores_the_expected_drop_of_b_times_1_minus_b():
    sc = make_scenario(cfg_n(1), 0)
    b = np.random.default_rng(0).uniform(0.05, 0.95, sc.grid.shape)
    g = PlanGrid(b, sc.lattice, sc.rates, 8, "variance")
    p = Pose(4, 4, 2)
    rng = np.random.default_rng(1)
    draws = [g.observe(g.new(rng), p, rng) for _ in range(400)]  # sampled world and observations
    assert np.mean(draws) == pytest.approx(g.gain(g.new(rng), p), rel=0.03)
    st = g.new(rng)
    before = variance(st.b).sum()
    drop = g.observe(st, p, rng)
    assert drop == pytest.approx((before - variance(st.b).sum()) * 64)
    with pytest.raises(ValueError):
        PlanGrid(b, sc.lattice, sc.rates, 8, "nope")


def test_variance_reward_plans_and_mh_refuses_it_with_g2():
    sc = make_scenario(cfg_n(2), 0)
    c = DecMCTSConfig(**{**CFG["mcts"], "reward": "variance"})
    acts = DecMCTS(sc.lattice, sc.rates, c, seed=0).plan([np.full(sc.grid.shape, 0.5)] * 2, sc.starts, [[1], [0]])
    assert all(a in sc.lattice.admissible(p) for a, p in zip(acts, sc.starts))
    with pytest.raises(ValueError):
        MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**{**CFG["mcts"], "reward": "variance"}), MHConfig())
    MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**{**CFG["mcts"], "reward": "variance"}), MHConfig(g2_weight=0))


# ------------------------------------------------------------------ root log
def test_root_log_records_the_tree_and_greedy_choice_and_changes_nothing():
    plain = one_run(cfg_n(2), IGD, "decmcts", 0, 3, 10)
    logged = one_run(cfg_n(2), IGD, "decmcts:log_root=true", 0, 3, 10)
    assert logged["mse"] == plain["mse"] and "root_agree" not in plain  # logging does not change the run
    for k in ("root_agree", "root_up", "greedy_up", "root_down", "greedy_down", "root_gap"):
        assert len(logged[k]) == 3 and all(0.0 <= v <= 1.0 for v in logged[k] if k != "root_gap")
    assert set(run_record(logged, 0, 42)["curves"]) >= {"root_agree", "root_gap"}
    sc = make_scenario(cfg_n(1), 0)
    dm = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(**{**CFG["mcts"], "log_root": True}), seed=0)
    a = dm.plan([np.full(sc.grid.shape, 0.5)], sc.starts, [[]])[0]
    (u,) = dm.root_log[0]
    assert u["action"] == a and u["greedy"] in u["ig"] and set(u["value"]) <= set(u["ig"])


# ------------------------------------------------------------------ planner seeds
def test_planner_seed_changes_only_the_planning_randomness():
    base = one_run(cfg_n(2), IGD, "decmcts", 0, 4, 10)
    assert one_run(cfg_n(2), IGD, "decmcts:planner_seed=0", 0, 4, 10)["mse"] == base["mse"]
    other = one_run(cfg_n(2), IGD, "decmcts:planner_seed=1", 0, 4, 10)
    assert np.array_equal(other["truth"], base["truth"]) and other["poses"][0][0] == base["poses"][0][0]
    assert other["poses"] != base["poses"]
    g0 = one_run(cfg_n(4), IGD, "greedy", 0, 6, 10)
    assert one_run(cfg_n(4), IGD, "greedy:planner_seed=0", 0, 6, 10)["poses"] == g0["poses"]
    assert one_run(cfg_n(4), IGD, "greedy:planner_seed=2", 0, 6, 10)["poses"] != g0["poses"]  # IGd's orders


# ------------------------------------------------------------------ phase switch
def test_switch_at_zero_coverage_is_greedy_and_never_switching_is_the_tree():
    g = one_run(cfg_n(2), IGD, "greedy", 0, 4, 10)
    s0 = one_run(cfg_n(2), IGD, "decmcts:switch_cov=0", 0, 4, 10)
    assert s0["poses"] == g["poses"] and s0["switch_step"] == 0
    tree = one_run(cfg_n(2), IGD, "decmcts", 0, 4, 10)
    never = one_run(cfg_n(2), IGD, "decmcts:switch_cov=1.1", 0, 4, 10)
    assert never["poses"] == tree["poses"] and never["switch_step"] is None


def test_switch_happens_once_coverage_is_reached():
    sc = make_scenario(cfg_n(2), 0)
    pl = make_planner("mhmcts", {**cfg_n(2), "mcts": {**CFG["mcts"], "switch_cov": 0.05}}, sc, 5, "igd")
    assert isinstance(pl, PhaseSwitch)
    b = [np.full(sc.grid.shape, 0.5)] * 2
    pl.plan(b, sc.starts, [[1], [0]])
    assert pl.switched_at is None
    seen = np.full(sc.grid.shape, 0.5)
    seen[:, : sc.grid.cols // 10] = 0.9
    pl.plan([seen, seen], sc.starts, [[1], [0]])
    assert pl.switched_at == 1


# ------------------------------------------------------------------ mhgreedy
@pytest.mark.parametrize("n, setting", [(1, IG), (2, IG), (4, IGD)])
def test_mhgreedy_without_g2_is_greedy(n, setting):
    g = one_run(cfg_n(n), setting, "greedy", 0, 6, 10)
    m = one_run(cfg_n(n), setting, "mhgreedy:g2_weight=0", 0, 6, 10)
    assert m["poses"] == g["poses"] and m["mse"] == g["mse"]


def test_mhgreedy_with_g2_and_budget_runs_and_heads_for_open_regions():
    for spec in ("mhgreedy", "mhgreedy:budget=true", "mhgreedy:hl_rollouts=0,stride=2"):
        r = one_run(cfg_n(2), IGD, spec, 0, 4, 10)
        assert len(r["mse"]) == 4 and len(r["g2_changed"]) == 4 and all(0 <= v <= 1 for v in r["g2_changed"])
    assert "g2_changed" not in one_run(cfg_n(2), IGD, "mhgreedy:g2_weight=0", 0, 4, 10)
    sc = make_scenario(cfg_n(1), 0)
    pl = make_planner("mhgreedy", {**cfg_n(1), "run": {"steps": 50}, "mh": {"tile": 12.5, "hl_rollouts": 0}}, sc)
    assert isinstance(pl, MHGreedy)
    b = np.full(sc.grid.shape, 0.5)
    b[:, : sc.grid.cols // 2] = 0.999  # west known, east open
    g2 = pl._g2(0, b, Pose(4, 4, 0), [], None)
    assert g2["right"] > g2["left"]  # moving east shortens the mission


def test_budget_counts_the_work_left_undone():
    sc = make_scenario(cfg_n(1), 0)
    r = Regions(sc.lattice, 8, 12.5)
    T = MissionTime(r, 1.0, 0.0)
    E = np.zeros(r.n)
    E[3] = 10.0  # 10 steps of work, 3 * per steps away
    d = 3 * r.per
    assert T(E, frozenset({3}), [(0, (0, 0))], budget=100) == 0.0
    assert T(E, frozenset({3}), [(0, (0, 0))], budget=d + 4) == 6.0  # 4 of 10 steps done
    assert T(E, frozenset({3}), [(0, (0, 0))], budget=d - 1) == 10.0
    assert T(E, frozenset(), [(0, (0, 0))], budget=5) == 0.0
    with pytest.raises(ValueError):
        MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**CFG["mcts"]), MHConfig(budget=True))


# ------------------------------------------------------------------ depot start
def test_depot_start_puts_every_uav_at_the_sw_corner():
    sc = make_scenario({**cfg_n(4), "agents": {"n": 4, "start": "depot"}}, 0)
    assert sc.starts == [Pose(0, 0, 0)] * 4
    r = one_run({**cfg_n(4), "agents": {"n": 4, "start": "depot"}}, IGD, "greedy", 0, 3, 10)
    assert len(r["mse"]) == 3
