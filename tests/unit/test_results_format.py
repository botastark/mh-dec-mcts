"""The result format: exact round trip, the old runs.json reader, and the statistics helpers."""

import json

import numpy as np

from mh_dec_mcts.experiment import (
    condition_id,
    curve_stats,
    load_arrays,
    load_results,
    paired_diff,
    run_record,
    save_results,
    se,
)


def fake_run(seed: int, steps: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    curve = lambda: rng.random(steps).tolist()
    return {"mse": curve(), "entropy": curve(), "team_entropy": curve(), "alt": curve(),
            "coverage": sorted(curve()), "coverage_paper": sorted(curve()), "plan": curve(),
            "misalignment": curve(), "overlap": curve(), "connected": curve(), "backfill": curve(),
            "stall": 3, "altitude": float(rng.random()), "balance": float(rng.random()),
            "cpu": 0.1 + rng.random(), "plan_per_step": float(rng.random()), "wall": 1.0}


def test_results_round_trip_exactly(tmp_path):
    runs = [run_record(fake_run(i), i, 42 + i) for i in range(3)]
    cond = {"id": condition_id("ig", 2, "BM", 5, None), "planner": "ig", "n": 2, "sharing": "BM",
            "range": 5, "setting": None, "steps": 5, "runs": runs}
    arrays = {"c0_r0_poses": np.arange(12).reshape(2, 2, 3)}
    save_results(tmp_path, {"config": "x.yaml"}, [cond], arrays)
    back = load_results(tmp_path)
    assert back["schema"] == 1 and back["conditions"] == [cond]  # floats compare with ==
    assert back["conditions"][0]["runs"][1]["curves"]["mse"] == runs[1]["curves"]["mse"]
    np.testing.assert_array_equal(load_arrays(tmp_path)["c0_r0_poses"], arrays["c0_r0_poses"])


def test_scalars_keep_the_old_runs_json_names():
    s = run_record(fake_run(0), 0, 0)["scalars"]
    assert {"avg_mse", "final_mse", "stall", "final_cov", "t80", "t95", "altitude", "balance",
            "cpu_per_step"} <= set(s)


def test_old_runs_json_reads_into_the_new_shape(tmp_path):
    old = {"meta": {"config": "configs/swarm.yaml", "setting": "IGd_BM_R5", "runs": 2, "steps": 4, "n": 1,
                    "trap": 15},
           "planners": {"greedy": {"runs": [{"avg_mse": 0.1, "final_mse": 0.05}, {"avg_mse": 0.2, "final_mse": 0.1}],
                                   "mean": {"mse": [0.25, 0.2, 0.1, 0.05]}}}}
    (tmp_path / "runs.json").write_text(json.dumps(old))
    d = load_results(tmp_path)
    (c,) = d["conditions"]
    assert c["planner"] == "greedy" and c["n"] == 1 and c["setting"] == "IGd_BM_R5"
    assert [r["scalars"]["avg_mse"] for r in c["runs"]] == [0.1, 0.2]
    assert all(r["curves"] is None for r in c["runs"])
    assert c["mean_curves"]["mse"] == [0.25, 0.2, 0.1, 0.05] and c["std_curves"] is None
    assert d["meta"]["trap"] == 15


def test_statistics_helpers():
    curves = [[1.0, 2.0], [3.0, 6.0]]
    mu, sd = curve_stats(curves)
    np.testing.assert_array_equal(mu, [2.0, 4.0])
    np.testing.assert_array_equal(sd, [1.0, 2.0])  # ddof 0
    assert se([1.0, 3.0]) == 1.0 / np.sqrt(2)
    np.testing.assert_array_equal(paired_diff([1.0, 2.0], [0.5, 2.5]), [0.5, -0.5])


def test_a_renamed_old_runs_json_is_found(tmp_path):
    old = {"meta": {"config": "c.yaml", "setting": "S", "runs": 1, "steps": 2, "n": 4, "trap": 15},
           "planners": {"greedy": {"runs": [{"avg_mse": 0.3}], "mean": {"mse": [0.25, 0.2]}}}}
    (tmp_path / "runs_rect_n4.json").write_text(json.dumps(old))
    assert load_results(tmp_path)["conditions"][0]["runs"][0]["scalars"]["avg_mse"] == 0.3
