"""scripts/run.py and its pieces: grid expansion, planner specs, compat ordering, the CLI."""

import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mh_dec_mcts import experiment
from mh_dec_mcts.experiment import parse_spec
from mh_dec_mcts.planning.mh_mcts import split_overrides
from mh_dec_mcts.scenario import load_config

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run

SWARM = str(REPO / "configs/swarm_rect.yaml")
FIG5 = str(REPO / "configs/pierdicca_fig5.yaml")


# ------------------------------------------------------------------ grid
def test_grid_is_planners_x_sharing_x_ranges_x_n_and_one_uav_collapses():
    cfg = load_config(FIG5)
    conds = run.grid(cfg, ["ig", "igd"], [1, 2], None, ["none", "BS"], [5, 16])
    two = [(c["planner"], c["sharing"], c["range"]) for c in conds if c["n"] == 2]
    assert two == [(p, s, r) for p in ("ig", "igd") for s in ("none", "BS") for r in (5, 16)]
    one = [(c["planner"], c["sharing"], c["range"]) for c in conds if c["n"] == 1]
    assert one == [("ig", "none", 5), ("igd", "none", 5)]  # nothing to share: one combination
    assert len({c["id"] for c in conds}) == len(conds) == 10


def test_one_uav_without_none_sharing_keeps_the_first_listed():
    conds = run.grid(load_config(FIG5), ["ig"], [1], None, ["BM", "BS"], [16, 5])
    assert [(c["sharing"], c["range"]) for c in conds] == [("BM", 16)]


def test_setting_fixes_sharing_range_and_greedy():
    cfg = load_config(SWARM)
    conds = run.grid(cfg, ["greedy", "decmcts"], [1, 4], "IGd_BM_R5", [], [])
    s = cfg["settings"]["IGd_BM_R5"]
    assert {(c["sharing"], c["range"], c["greedy"], c["setting"]) for c in conds} == \
        {(s["sharing"], s["range"], "igd", "IGd_BM_R5")}
    assert [(c["planner"], c["n"]) for c in conds] == [("greedy", 1), ("greedy", 4), ("decmcts", 1), ("decmcts", 4)]


def test_greedy_needs_a_setting():
    with pytest.raises(SystemExit):
        run.grid(load_config(FIG5), ["greedy"], [2], None, ["none"], [5])


# ------------------------------------------------------------------ specs
@pytest.mark.parametrize("spec, name, over", [
    ("ig", "ig", {}), ("random", "random", {}), ("greedy", "greedy", {}),
    ("decmcts:horizon=8,rollout=greedy", "decmcts", {"horizon": 8, "rollout": "greedy"}),
    ("mhmcts:horizon=4,tile=12.5,g2_weight=0", "mhmcts", {"horizon": 4, "tile": 12.5, "g2_weight": 0}),
])
def test_parse_spec(spec, name, over):
    assert parse_spec(spec) == (name, over)
    assert experiment.parse is parse_spec  # old name kept


def test_mh_keys_go_to_mh_the_rest_to_mcts(monkeypatch):
    """one_run builds the planner with the spec's keys routed to their config sections."""
    seen = {}

    class Stop(Exception):
        pass

    def fake_make_planner(name, cfg, sc, range_):
        seen.update(name=name, mcts=cfg["mcts"], mh=cfg["mh"])
        raise Stop

    monkeypatch.setattr(experiment, "make_planner", fake_make_planner)
    cfg = load_config(SWARM, agents={"n": 1})
    with pytest.raises(Stop):
        experiment.one_run(cfg, {"sharing": "none", "range": 5, "greedy": "igd"},
                           "mhmcts:horizon=2,rollout=random,tile=12.5,hl_rollouts=0", 0, 3, 10)
    assert seen["name"] == "mhmcts"
    assert seen["mcts"]["horizon"] == 2 and seen["mcts"]["rollout"] == "random"
    assert seen["mh"]["tile"] == 12.5 and seen["mh"]["hl_rollouts"] == 0
    assert "tile" not in seen["mcts"] and "horizon" not in seen["mh"]
    assert split_overrides({"horizon": 2, "tile": 1.0}) == ({"horizon": 2}, {"tile": 1.0})


# ------------------------------------------------------------------ compat ordering
def _compat_runs(n_runs: int) -> list:
    cfg = run.condition_cfg(FIG5, {}, {"n": 2, "sharing": "BM", "range": 5})
    setting = {"sharing": "BM", "range": 5, "greedy": None}
    rs = run.job_compat(cfg, setting, "igd", n_runs, 4, 10, 0.01, False)
    return [(r["mse"], [[(p.ix, p.iy, p.k) for p in path] for path in r["poses"]]) for r in rs]


def test_compat_runs_are_built_in_order_with_one_streams_and_reproduce_in_a_fresh_process():
    with mp.get_context("spawn").Pool(1) as pool:  # a fresh interpreter each time
        a = pool.apply(_compat_runs, (3,))
    with mp.get_context("spawn").Pool(1) as pool:
        b = pool.apply(_compat_runs, (3,))
    assert a == b
    # Pierdicca's generators carry over: run 1 is not run 1 built on its own (fresh streams)
    cfg = run.condition_cfg(FIG5, {}, {"n": 2, "sharing": "BM", "range": 5})
    alone = experiment.one_run(cfg, {"sharing": "BM", "range": 5, "greedy": None}, "igd", 1, 4, 10)
    assert alone["mse"] != a[1][0]


# ------------------------------------------------------------------ CLI
def test_run_py_smoke(tmp_path):
    subprocess.run([sys.executable, str(REPO / "scripts/run.py"), SWARM, "--setting", "IGd_BM_R5", "--n", "1",
                    "--planners", "greedy", "--runs", "1", "--steps", "3", "--jobs", "1", "--out", str(tmp_path),
                    "--tag", "t"], check=True, capture_output=True, cwd=REPO)
    d = json.loads((tmp_path / "t/results.json").read_text())
    assert d["schema"] == 1 and len(d["conditions"]) == 1
    (r,) = d["conditions"][0]["runs"]
    assert len(r["curves"]["mse"]) == 3 and r["run_seed"] == d["meta"]["config_resolved"]["seed"]
    assert "avg MSE" in (tmp_path / "t/results.txt").read_text()
    with np.load(tmp_path / "t/results.npz") as z:  # always saved: everything the figures need
        assert {"c0_r0_poses", "c0_r0_first", "c0_r0_uav_mse", "c0_r0_uav_alt", "truth_r0"} <= set(z.files)
        assert z["c0_r0_uav_mse"].shape == (3, 1) and "c0_r0_beliefs" not in z.files  # beliefs: --save-beliefs
