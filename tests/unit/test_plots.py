"""plots.paired.merge: planners run in separate folders paired on the same runs."""

import copy

import pytest

from mh_dec_mcts.plots.paired import compare, merge


def result(planners, seed=42, runs=3, steps=5):
    conds = [{"id": f"{p}|n=4", "planner": p, "n": 4, "sharing": "BM", "range": 5, "setting": "S", "steps": steps,
              "runs": [{"run": i, "run_seed": seed + i, "scalars": {"t95": 3}, "curves": {"mse": [0.1] * steps}}
                       for i in range(runs)]} for p in planners]
    return {"meta": {"config": "c", "config_resolved": {"name": "c"}, "trap": 15}, "conditions": conds}


def test_merge_pairs_conditions_of_the_same_runs_and_drops_duplicates():
    m = merge([result(["greedy", "a"]), result(["greedy", "b"])])
    assert [c["planner"] for c in m["conditions"]] == ["greedy", "a", "b"]
    (block,) = compare(m, ref="greedy")
    assert [c["planner"] for c in block["others"]] == ["a", "b"]


@pytest.mark.parametrize("other", [result(["greedy", "b"], seed=1042), result(["b"], runs=2), result(["b"], steps=4)])
def test_merge_refuses_results_of_other_runs(other):
    with pytest.raises(ValueError):
        merge([result(["greedy", "a"]), copy.deepcopy(other)])
