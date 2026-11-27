"""MH-MCTS, step 1: regions; step 2: mission time (g2); step 3: low-level planner with g2;
step 5: the high-level planner (HLP)."""

from typing import Any

import numpy as np

from mh_dec_mcts.planning.dec_mcts import DecMCTS, DecMCTSConfig
from mh_dec_mcts.planning.mh_mcts import (MHConfig, MHMCTS, MissionTime, Regions, RegionTree, look_rate,
                                          make_mh_mcts, split_overrides)
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



def test_an_order_is_followed_and_closed_regions_are_skipped():
    r, T = mission(lam=1e12)  # pure travel
    E = np.ones(r.n)
    op = frozenset({1, 3, 7})
    # nearest-first from the SW corner: 1, 3, 7; the order 7, 1, 3 flies to the far corner first
    nearest = T(E, op, [(0, (0, 0))])
    assert np.isclose(T(E, op, [(0, (0, 0))], [(1, 3, 7)]), nearest)
    far_first = r.dist((0, 0), 7) + r.dist(r.center[7], 1) + r.dist(r.center[1], 3)
    assert np.isclose(T(E, op, [(0, (0, 0))], [(7, 1, 3)]), far_first) and far_first > nearest
    # regions of the order that are closed are skipped; open ones missing from it come last, nearest-first
    assert np.isclose(T(E, op, [(0, (0, 0))], [(5, 7, 6)]), r.dist((0, 0), 7) + r.dist(r.center[7], 3) + r.dist(r.center[3], 1))


def test_orders_divide_the_regions_between_uavs():
    r, T = mission(lam=1.0)
    E = np.zeros(r.n)
    E[[0, 3]] = 8.0  # 8 steps of work each
    west, east = (0, 0), (3 * r.per + r.per, 0)  # SW corner, SE corner
    # each UAV takes the region at its own end: both finish after 8 steps of work
    split = T(E, frozenset({0, 3}), [(0, west), (0, east)], [(0, 3), (3, 0)])
    assert split == 8.0
    # both orders start at the far region: the UAVs cross the field
    crossed = T(E, frozenset({0, 3}), [(0, west), (0, east)], [(3, 0), (0, 3)])
    assert crossed > split


def test_single_uav_tours_are_cached_per_order():
    r, T = mission(lam=1.0)
    E = np.ones(r.n)
    op = frozenset(range(r.n))
    a = T(E, op, [(0, (0, 0))])
    assert len(T._tours) == 1 and T(E * 2, op, [(0, (0, 0))]) == a + float(np.sum(T.works(E * 2) - T.works(E)))
    T(E, op, [(0, (0, 0))], [(7,)])  # another order: its own entry
    assert len(T._tours) == 2


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


def test_team_run_with_a_coverage_adaptive_stride():
    """The stride drops from 8 to 4 once a UAV has seen 5 % of the field: regions and mission time
    switch with it, HL trees and orders carry over (region ids do not depend on the stride)."""
    cfg = {**CFG, "agents": {"n": 2, "start": "legacy"}, "mcts": {**FAST, "stride_fine": 4, "stride_cov": 0.05},
           "mh": {"tile": 12.5}}
    sc = make_scenario(cfg, 0)
    mh = make_mh_mcts(cfg, sc)
    res = run_multi(sc, mh, 4, "adaptive", "BM", 5)
    assert all(len(p) == 5 for p in res.poses)
    assert mh.last_strides == [4, 4] and set(mh._times) == {8, 4}
    assert mh.time_at(4).reg.n == mh.time_at(8).reg.n
    assert abs(mh.time_at(4).lam / mh.time_at(8).lam - 1) < 0.25  # bits per look, either grid


# ------------------------------------------------------------------ step 4: wiring
def test_spec_overrides_go_to_their_config_sections():
    mcts, mh = split_overrides({"horizon": 4, "rollout": "greedy", "tile": 12.5, "g2_weight": 0.0})
    assert mcts == {"horizon": 4, "rollout": "greedy"} and mh == {"tile": 12.5, "g2_weight": 0.0}
    assert split_overrides({}) == ({}, {})


# ------------------------------------------------------------------ step 5: high-level planner
def test_hl_tree_finds_a_shorter_tour_than_nearest_first():
    """Bottom row of 4 tiles, UAV just inside tile 1: nearest-first goes 1, 0 (tie with 2, lower
    id), back across to 2, 3: 10 steps. Starting with 0 then sweeping east is 7."""
    r, T = mission(lam=1e12)  # work ~ 0: the order only changes travel
    E, op, start = np.ones(r.n), frozenset({0, 1, 2, 3}), (5, 0)
    assert np.isclose(T(E, op, [(0, start)]), 10)
    hl = RegionTree(op, gamma=1.0, cp=0.5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        node = hl.select_expand(rng)
        order = hl.complete(node, rng, r, start)
        if not node.seq:
            node.seq = order
        hl.backprop(node, -T(E, op, [(0, start)], [order]))
    best = hl.exploit_sequence()
    assert sorted(best) == [0, 1, 2, 3] and np.isclose(T(E, op, [(0, start)], [best]), 7)


def test_hl_tree_completion_policies():
    r, _ = mission()
    hl = RegionTree(frozenset({0, 1, 2, 3}), 1.0, 0.5)
    rng = np.random.default_rng(1)
    assert hl.complete(hl.root, rng, r, (0, 0)) == (0, 1, 2, 3)  # nearest-first from the SW corner
    node = hl.select_expand(rng)  # one region chosen by the tree, the rest nearest-first from it
    assert hl.complete(node, rng, r, (0, 0))[0] == node.pose
    assert sorted(hl.complete(node, rng, None, (0, 0))) == [0, 1, 2, 3]  # random completion


def test_without_hl_rollouts_it_is_the_g2_only_planner():
    """hl_rollouts = 0: no HL tree, every UAV nearest-first; an HL tree changes the orders only."""
    sc = make_scenario({**CFG, "agents": {"n": 2, "start": "legacy"}}, 0)
    beliefs = [np.full(sc.grid.shape, 0.5)] * 2
    off = MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST), MHConfig(tile=12.5, hl_rollouts=0), seed=0)
    off.plan(beliefs, sc.starts, [[1], [0]])
    assert off.hl_trees == {} and off.last_hl == [None, None]
    on = MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST), MHConfig(tile=12.5), seed=0)
    on.plan(beliefs, sc.starts, [[1], [0]])
    for r in range(2):
        assert sorted(on.last_hl[r]) == sorted(on.last_open[r])  # a full order of the open regions
        assert on._hl_known[1 - r][r] == on.last_hl[r]  # ... received by the teammate in range


def test_hl_tree_is_kept_across_steps_while_regions_only_close():
    sc = make_scenario(CFG, 0)
    b = np.full(sc.grid.shape, 0.5)
    mh = MHMCTS(sc.lattice, sc.rates, DecMCTSConfig(**FAST), MHConfig(tile=12.5), seed=0)
    mh.plan([b], [sc.starts[0]], [[]])
    tree, size = mh.hl_trees[0], len(mh.hl_trees[0].nodes)
    mh.plan([b], [sc.starts[0]], [[]])
    assert mh.hl_trees[0] is tree and len(tree.nodes) > size  # same tree, grown further
    b2 = b.copy()
    b2[: b.shape[0] // 2, : b.shape[1] // 4] = 0.001  # one tile (north-west) becomes known
    mh.plan([b2], [sc.starts[0]], [[]])
    assert mh.hl_trees[0] is tree and tree.open == mh.last_open[0] == tree.key - {4}  # kept, tile dropped
    assert 4 not in mh.last_hl[0] and sorted(mh.last_hl[0]) == sorted(tree.open)
    b3 = b.copy()
    b3[:, : 3 * b.shape[1] // 4] = 0.001  # 6 of 8 tiles known: fewer than half left
    mh.plan([b3], [sc.starts[0]], [[]])
    assert mh.hl_trees[0] is not tree and mh.hl_trees[0].key == mh.last_open[0]  # rebuilt


def test_closed_regions_leave_the_hl_search():
    r, T = mission(lam=1e12)
    op = frozenset({0, 1, 2, 3})
    hl = RegionTree(op, 1.0, 0.5)
    rng = np.random.default_rng(0)
    for _ in range(40):
        node = hl.select_expand(rng)
        order = hl.complete(node, rng, r, (0, 0))
        node.seq = node.seq or order
        hl.backprop(node, -T(np.ones(r.n), op, [(0, (0, 0))], [order]))
    hl.close(frozenset({1, 3}))
    for _ in range(40):
        node = hl.select_expand(rng)
        assert set(hl.prefix(node)) <= {1, 3}  # never descends into closed regions
        assert sorted(hl.complete(node, rng, r, (0, 0))) == [1, 3]
        node.seq = node.seq or hl.complete(node, rng, r, (0, 0))
        hl.backprop(node, 0.0)
    assert sorted(hl.exploit_sequence()) == [1, 3]
