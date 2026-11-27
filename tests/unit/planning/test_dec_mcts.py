"""Dec-MCTS: planning objective, D-UCT tree, Alg. 3, team planning, wiring. Fast settings only."""

from typing import Any

import numpy as np
import pytest

from mh_dec_mcts.comms.network import neighbours
from mh_dec_mcts.planning.dec_mcts import DecMCTS, DecMCTSConfig, Tree
from mh_dec_mcts.planning.information import info_gain
from mh_dec_mcts.planning.objective import PlanGrid
from mh_dec_mcts.planning.planners import PLANNERS, make_planner
from mh_dec_mcts.scenario import make_scenario
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.sim.multi import run_multi

CFG: dict[str, Any] = {"name": "t", "grid": {"size_x": 25, "size_y": 25, "cell": 0.125}, "lattice": {"xy_step": None},
                       "field": {"type": "gaussian", "r": 3}, "agents": {"n": 2, "start": "random"},
                       "run": {"weights": "adaptive"},
                       "mcts": {"horizon": 3, "iterations": 6, "rollouts": 5, "n_seq": 5, "resample_every": 3, "stride": 8}}
FAST = DecMCTSConfig(**CFG["mcts"])


def scen(n=2, run=0):
    return make_scenario({**CFG, "agents": {"n": n, "start": "random"}}, run)


# ------------------------------------------------------------------ objective
def test_footprints_on_the_strided_grid():
    sc = scen()
    pg = PlanGrid(np.full(sc.grid.shape, 0.5), sc.lattice, sc.rates, 4)
    for p in [Pose(0, 0, 0), Pose(8, 8, 3), Pose(16, 5, 5)]:
        i0, i1, j0, j1 = sc.lattice.footprint(p)
        a0, a1, b0, b1 = pg.bounds(p)
        assert (a0, a1, b0, b1) == (-(-i0 // 4), -(-i1 // 4), -(-j0 // 4), -(-j1 // 4))
        assert (a1 - a0) * 4 == pytest.approx(i1 - i0, abs=4)


def test_gain_is_the_bayes_information_gain():
    sc = scen()
    b = np.random.default_rng(0).uniform(0.05, 0.95, sc.grid.shape)
    pg = PlanGrid(b, sc.lattice, sc.rates, 1)
    p = Pose(5, 7, 2)
    i0, i1, j0, j1 = sc.lattice.footprint(p)
    ref = info_gain(b[i0:i1, j0:j1], *sc.rates.at(2)).sum()
    assert pg.gain(pg.new(np.random.default_rng(0)), p) == pytest.approx(ref, rel=1e-5)  # legacy info_gain rounds rates


def test_one_look_drops_entropy_by_the_ig_on_average():
    """E[entropy drop of one sampled look] = expected IG (Bayes)."""
    sc = scen()
    b = np.random.default_rng(1).uniform(0.05, 0.95, sc.grid.shape)
    pg = PlanGrid(b, sc.lattice, sc.rates, 2)
    p, rng = Pose(6, 6, 3), np.random.default_rng(2)
    drops = [pg.observe(pg.new(rng), p, rng) for _ in range(200)]
    assert np.mean(drops) == pytest.approx(pg.gain(pg.new(rng), p), rel=0.02)


def test_teammate_looks_lower_my_gain():
    sc = scen()
    pg = PlanGrid(np.full(sc.grid.shape, 0.5), sc.lattice, sc.rates, 4)
    mine = [Pose(8, 8, 2), Pose(8, 9, 2)]
    rng = np.random.default_rng(0)
    alone = np.mean([pg.sequence_gain(pg.new(rng), mine, rng) for _ in range(5)])
    after_same = np.mean([pg.sequence_gain(pg.with_others([mine], rng), mine, rng) for _ in range(5)])
    after_far = np.mean([pg.sequence_gain(pg.with_others([[Pose(0, 0, 0)]], rng), mine, rng) for _ in range(5)])
    assert after_same < 0.8 * alone
    assert after_far == pytest.approx(alone, rel=0.03)


# ------------------------------------------------------------------ D-UCT tree
def test_discounted_counts():
    sc = scen()
    tree = Tree(sc.lattice, Pose(8, 8, 2), DecMCTSConfig(gamma=0.5, horizon=2))
    rng = np.random.default_rng(0)
    a = tree.select_expand(rng)
    tree.backprop(a, 1.0)
    b = tree.select_expand(rng)  # root still has untried actions: a sibling of a
    assert b.parent is tree.root and b is not a
    tree.backprop(b, 3.0)
    assert a.n == pytest.approx(0.5) and a.w == pytest.approx(0.5)  # discounted once
    assert b.n == pytest.approx(1.0) and b.mean == pytest.approx(3.0)
    assert tree.f_max == 3.0


def test_best_sequences_are_full_length_and_start_next_to_the_root():
    sc = scen()
    root = Pose(8, 8, 2)
    dm = DecMCTS(sc.lattice, sc.rates, FAST, seed=0)
    dm.choose(np.full(sc.grid.shape, 0.5), root)
    seqs, q = dm.last_q[0]
    assert 0 < len(seqs) <= FAST.n_seq and q.sum() == pytest.approx(1.0)
    for s in seqs:
        assert len(s) == FAST.horizon
        assert s[0] in [sc.lattice.move(root, a) for a in sc.lattice.admissible(root)]


# ------------------------------------------------------------------ Alg. 3
def test_distribution_moves_to_the_better_sequence():
    sc = scen()
    b = np.full(sc.grid.shape, 0.5)
    b[:, : sc.grid.cols // 2] = 0.999  # west half already known
    pg = PlanGrid(b, sc.lattice, sc.rates, 8)
    west = tuple(Pose(2, 8, 1) for _ in range(3))
    east = tuple(Pose(14, 8, 1) for _ in range(3))
    dm = DecMCTS(sc.lattice, sc.rates, FAST, seed=0)
    q = np.array([0.5, 0.5])
    for _ in range(10):
        q = dm._update_q(pg, [west, east], q, {}, {}, 0.1, 1e4)
    assert q[1] > 0.6


# ------------------------------------------------------------------ planning stride
def test_stride_fixed_or_switched_by_own_coverage():
    sc = scen()
    b = np.full(sc.grid.shape, 0.5)
    assert DecMCTS(sc.lattice, sc.rates, DecMCTSConfig()).stride_for(b) == 1  # default: full grid, as greedy
    adapt = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(stride=8, stride_fine=2, stride_cov=0.5))
    assert adapt.stride_for(b) == 8  # nothing seen: coarse
    b[: b.shape[0] // 2 - 1] = 0.3
    assert adapt.stride_for(b) == 8  # just under half seen
    b[: b.shape[0] // 2 + 1] = 0.3
    assert adapt.stride_for(b) == 2  # past half: fine


def test_each_uav_plans_at_its_own_stride():
    sc = scen()
    seen, unseen = np.full(sc.grid.shape, 0.2), np.full(sc.grid.shape, 0.5)
    dec = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(**{**CFG["mcts"], "stride_fine": 4}), seed=0)
    acts = dec.plan([seen, unseen], sc.starts, [[1], [0]])
    assert dec.last_strides == [4, 8] and len(acts) == 2


# ------------------------------------------------------------------ team planning
def test_actions_admissible_and_reproducible():
    sc = scen(3)
    beliefs = [np.full(sc.grid.shape, 0.5)] * 3
    nb = neighbours(sc.starts, None)
    a1 = DecMCTS(sc.lattice, sc.rates, FAST, seed=1).plan(beliefs, sc.starts, nb)
    a2 = DecMCTS(sc.lattice, sc.rates, FAST, seed=1).plan(beliefs, sc.starts, nb)
    assert a1 == a2
    for p, a in zip(sc.starts, a1):
        assert a in sc.lattice.admissible(p)


def test_goes_where_the_map_is_unknown():
    sc = scen(1)
    b = np.full(sc.grid.shape, 0.5)
    b[:, : sc.grid.cols // 2] = 0.999  # west known, east unknown
    p = Pose(8, 8, 1)
    acts = [DecMCTS(sc.lattice, sc.rates, FAST, seed=s).choose(b, p) for s in range(5)]
    assert "left" not in acts and "right" in acts  # right = +x = east ("up" also sees more of it)


def test_lost_messages_equal_no_communication():
    sc = scen(2)
    beliefs = [np.full(sc.grid.shape, 0.5)] * 2
    cfg_drop = DecMCTSConfig(**{**CFG["mcts"], "drop": 1.0})
    silent = DecMCTS(sc.lattice, sc.rates, FAST, seed=4)
    silent.plan(beliefs, sc.starts, [[], []])
    lost = DecMCTS(sc.lattice, sc.rates, cfg_drop, seed=4)  # message loss has its own generator
    lost.plan(beliefs, sc.starts, [[1], [0]])
    for (s1, q1), (s2, q2) in zip(silent.last_q, lost.last_q):
        assert s1 == s2
        np.testing.assert_allclose(q1, q2)


# ------------------------------------------------------------------ wiring
@pytest.mark.parametrize("name", PLANNERS)
def test_every_planner_runs_alone_and_in_a_team(name):
    sc = scen(1)  # one UAV: the loop behind Pierdicca's Figs. 3 and 4 (the old sim.single, identical)
    res = run_multi(sc, make_planner(name, CFG, sc), 3, "adaptive")
    assert len(res.poses) == 1 and len(res.poses[0]) == 4
    sc = scen(3)
    res = run_multi(sc, make_planner(name, CFG, sc, 5), 3, "adaptive", "BM", 5)
    assert all(len(p) == 4 for p in res.poses)
    with pytest.raises(ValueError):
        make_planner("nope", CFG, sc)


def test_share_plans_off_means_planning_alone():
    sc = scen(2)
    beliefs = [np.full(sc.grid.shape, 0.5)] * 2
    alone = DecMCTS(sc.lattice, sc.rates, FAST, seed=5)
    alone.plan(beliefs, sc.starts, [[], []])
    off = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(**{**CFG["mcts"], "share_plans": False}), seed=5)
    off.plan(beliefs, sc.starts, [[1], [0]])
    for (s1, q1), (s2, q2) in zip(alone.last_q, off.last_q):
        assert s1 == s2
        np.testing.assert_allclose(q1, q2)


def test_single_intent_variant():
    """MH-MCTS paper variant: one intent per agent (the exploitation path), q = [1]."""
    sc = scen(2)
    beliefs = [np.full(sc.grid.shape, 0.5)] * 2
    dm = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(**{**CFG["mcts"], "variant": "single"}), seed=0)
    acts = dm.plan(beliefs, sc.starts, [[1], [0]])
    for r, (seqs, q) in enumerate(dm.last_q):
        assert len(seqs) == 1 and q.tolist() == [1.0] and len(seqs[0]) == FAST.horizon
        assert seqs[0] == dm.last_trees[r].exploit_sequence()
        assert sc.lattice.move(sc.starts[r], acts[r]) == seqs[0][0]
    with pytest.raises(ValueError):
        DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(variant="x")).plan(beliefs, sc.starts, [[1], [0]])
