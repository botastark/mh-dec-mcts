"""MH-MCTS, step 1: regions; step 2: mission time (g2); step 3: low-level planner with g2."""

from typing import Any

import numpy as np

from mh_dec_mcts.planning.dec_mcts import DecMCTS, DecMCTSConfig
from mh_dec_mcts.planning.mh_mcts import MHConfig, MHMCTS, MissionTime, Regions, look_rate, make_mh_mcts, split_overrides
from mh_dec_mcts.planning.objective import PlanGrid
from mh_dec_mcts.scenario import make_scenario
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.sim.multi import run_multi

CFG = {"name": "t", "grid": {"size_x": 50, "size_y": 25, "cell": 0.125}, "lattice": {"xy_step": 3.125},
       "field": {"type": "gaussian", "r": 5, "ref": 50.0}, "agents": {"n": 1, "start": "legacy"}}


def regions(tile=12.5, stride=8):
    sc = make_scenario(CFG, 0)
    return sc, Regions(sc.lattice, stride, tile)


def test_tiles_cover_the_field():
    sc, r = regions()
    assert (r.nrx, r.nry, r.per) == (4, 2, 4)  # 50 x 25 m in 12.5 m tiles of 4 waypoint steps
    assert r.label.size == (sc.grid.rows // 8) * (sc.grid.cols // 8) and set(r.label) == set(range(8))
    assert r.cells.max() - r.cells.min() <= 0.2 * r.cells.max()  # near-equal on the stride grid
    assert r.box[0] == (0, 4, 0, 4) and r.box[7] == (12, 16, 4, 8)  # SW corner, NE corner


def test_region_ids_run_east_then_north():
    sc, r = regions()
    lab = r.label.reshape(sc.grid.rows // 8, sc.grid.cols // 8)
    assert lab[-1, 0] == 0 and lab[-1, -1] == r.nrx - 1  # bottom row (south): 0 .. nrx-1
    assert lab[0, 0] == r.nrx * (r.nry - 1) and lab[0, -1] == r.n - 1  # top row (north)


def test_entropy_per_region():
    sc, r = regions()
    b = np.full(sc.grid.shape, 0.5)
    sub = PlanGrid(b, sc.lattice, sc.rates, 8).base
    np.testing.assert_allclose(r.mean_entropy(sub), 1.0)  # unknown field: 1 bit per cell
    assert np.isclose(r.entropy(sub).sum(), sub.size * 64)  # full-resolution bits
    sub = sub.copy()
    sub[r.label.reshape(sub.shape) == 3] = 0.999  # region 3 (south-east) nearly known
    m = r.mean_entropy(sub)
    assert m[3] < 0.02 and np.allclose(np.delete(m, 3), 1.0)


def test_distance_to_a_region():
    _, r = regions()
    assert r.dist(r.center[5], 5) == 0  # inside
    assert r.dist((0, 0), 3) == 3 * r.per  # SW corner to the SE tile: 12 steps east
    assert r.dist((0, 0), 7) == 3 * r.per + r.per  # ... and 4 north for the NE tile


# ------------------------------------------------------------------ step 2: mission time
def mission(lam=1.0, done=0.0):
    _, r = regions()  # 4 x 2 regions of 4 waypoint steps
    return r, MissionTime(r, lam, done)


def test_one_uav_travel_plus_work():
    r, T = mission(lam=1.0)
    E = np.zeros(r.n)
    E[3] = 10.0  # 10 bits at 1 bit / step: 10 steps of work
    assert T(E, frozenset({3}), [(2, (0, 0))]) == 2 + 3 * r.per + 10  # start time + travel + work


def test_closing_level_counts_only_the_work_left():
    r, T = mission(lam=1.0, done=0.5)
    full = r.cells[3] * r.scale  # bits of region 3 when unknown
    E = np.zeros(r.n)
    E[3] = full
    assert np.isclose(T.work(E, 3), 0.5 * full)  # down to 0.5 bit per cell, not to 0
    E[3] = 0.4 * full
    assert T.work(E, 3) == 0.0


def test_regions_are_visited_nearest_first():
    r, T = mission(lam=1e12)  # work ~ 0: pure travel
    E = np.ones(r.n)
    t = T(E, frozenset({1, 3}), [(0, (0, 0))])
    assert np.isclose(t, r.per + (r.dist(r.center[1], 3)))  # to 1 (4 steps), then from its centre to 3


def test_two_uavs_the_earlier_one_takes_the_region_and_the_mean_is_reported():
    r, T = mission(lam=1.0)
    E = np.zeros(r.n)
    E[3] = 10.0
    t = T(E, frozenset({3}), [(2, (0, 0)), (0, r.center[3])])
    assert t == (2 + 0 + 10) / 2  # UAV 2 (time 0, inside) works it; UAV 1 stays at time 2


def test_a_plan_ending_closer_to_the_open_region_shortens_the_mission():
    r, T = mission(lam=1.0)
    E = np.full(r.n, 100.0)
    assert T(E, frozenset({3}), [(3, (3 * r.per - 3, 0))]) < T(E, frozenset({3}), [(3, (0, 0))])


def test_open_regions_and_the_look_rate():
    sc, r = regions()
    sub = PlanGrid(np.full(sc.grid.shape, 0.5), sc.lattice, sc.rates, 8).base.copy()
    sub[r.label.reshape(sub.shape) == 3] = 0.999
    T = MissionTime(r, 1.0, 0.3)
    assert T.open(sub) == frozenset(range(r.n)) - {3}
    rates = [look_rate(sc.lattice, sc.rates, 8, k) for k in range(sc.lattice.n_alt)]
    assert all(x > 0 for x in rates) and rates[0] < max(rates)  # low looks see little ground


# ------------------------------------------------------------------ step 3: low-level planner with g2
FAST: dict[str, Any] = {"horizon": 3, "iterations": 5, "rollouts": 6, "stride": 8}


def test_without_g2_it_is_dec_mcts():
    sc = make_scenario({**CFG, "agents": {"n": 2, "start": "legacy"}}, 0)
    beliefs = [np.full(sc.grid.shape, 0.5)] * 2
    for seed in range(3):
        dec = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST, variant="single"), seed=seed)
        mh = MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST), MHConfig(g2_weight=0.0), seed=seed)
        assert mh.plan(beliefs, sc.starts, [[1], [0]]) == dec.plan(beliefs, sc.starts, [[1], [0]])
        assert [t.exploit_sequence() for t in mh.last_trees] == [t.exploit_sequence() for t in dec.last_trees]


def test_g2_pulls_a_uav_out_of_an_explored_area():
    """West half known, east half unknown, UAV low at the west edge: nothing to gain within a
    3-step horizon, but g2 rewards heading east. Plain Dec-MCTS has no reason to."""
    sc = make_scenario(CFG, 0)
    b = np.full(sc.grid.shape, 0.5)
    b[:, : b.shape[1] // 2] = 0.001
    pose = Pose(0, sc.lattice.ny // 2, 0)
    mh_moves, dec_moves = [], []
    for seed in range(10):
        mh = MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST), MHConfig(tile=12.5), seed=seed)
        mh_moves += mh.plan([b], [pose], [[]])
        assert mh.last_open[0] == frozenset({2, 3, 6, 7})  # the east half, in 12.5 m tiles
        dec = DecMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST, variant="single"), seed=seed)
        dec_moves += dec.plan([b], [pose], [[]])
    assert mh_moves.count("right") == 10
    assert dec_moves.count("right") < 10


def test_team_run_with_the_scenario_settings():
    cfg = {**CFG, "agents": {"n": 2, "start": "legacy"}, "mcts": FAST, "mh": {"tile": 12.5}}
    sc = make_scenario(cfg, 0)
    mh = make_mh_mcts(cfg, sc)
    res = run_multi(sc, mh, 3, "adaptive", "BM", 5)
    assert all(len(p) == 4 for p in res.poses)  # start + 3 moves, every move admissible
    assert all(len(o) == mh.reg.n for o in mh.last_open)  # nothing finished after 3 steps


# ------------------------------------------------------------------ step 4: wiring
def test_spec_overrides_go_to_their_config_sections():
    mcts, mh = split_overrides({"horizon": 4, "rollout": "greedy", "tile": 12.5, "g2_weight": 0.0})
    assert mcts == {"horizon": 4, "rollout": "greedy"} and mh == {"tile": 12.5, "g2_weight": 0.0}
    assert split_overrides({}) == ({}, {})
