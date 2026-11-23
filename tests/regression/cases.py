"""Regression cases for the runner refactor: small versions of every experiment the scripts run.

collect(tmp) runs every case through the current code and returns {case name: canonical data},
plain JSON-able values (floats kept exactly, never rounded). make_baseline.py stores that once;
test_baseline.py recomputes it and compares by exact equality. As scripts are replaced, only the
runners below change: the canonical data of each case must stay the same.

Timings (CPU / wall seconds, planning time) differ from run to run and are left out.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
TIMING = {"cpu", "wall", "plan", "plan_per_step", "cpu_per_step", "s_per_step"}

MYOPIA_SPECS = ["greedy", "decmcts:horizon=4,rollout=greedy", "mhmcts:horizon=4,rollout=greedy",
                "mhmcts:horizon=4,rollout=greedy,hl_rollouts=0,tile=12.5"]
MYOPIA_CASES = {  # name: (config, setting, n, runs, steps)
    "myopia_rect_n1": ("configs/swarm_rect.yaml", "IGd_BM_R5", 1, 2, 30),
    "myopia_rect_n4": ("configs/swarm_rect.yaml", "IGd_BM_R5", 4, 2, 30),
    "myopia_rect_n4_bs": ("configs/swarm_rect.yaml", "IG_BS_Rinf", 4, 2, 30),
}
SINGLE_CASES = {  # name: (config, extra args, runs, steps)
    "single_paper": ("configs/paper.yaml", [], 2, 30),
    "single_fig3": ("configs/pierdicca_fig3.yaml", [], 2, 30),
    "single_fig4": ("configs/pierdicca_fig4.yaml", ["--planners", "ig", "--weights", "adaptive", "--r", "2"], 2, 30),
}
MULTI_CASE = ("configs/pierdicca_fig5.yaml", ["ig", "igd"], ["none", "BS", "BM"], [5, 16], [1, 2, 4], 2, 20)
TUNE_CASE = ("configs/swarm.yaml", ["horizon", "2", "4"], 2, 20)
SWEEP_CASE = (["square_n1"], [2, 4], 2, 20)


def strip(x):
    """Drop timing keys anywhere in a nested structure; tuples become lists (as in JSON)."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def sh(args, cwd=REPO):
    subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, check=True, stdout=subprocess.DEVNULL)


# ------------------------------------------------------------------ runners
# single / multi / myopia cases: scripts/run.py (T3), converted to what the old scripts saved.
def run_py(tmp: Path, config, args: list[str]) -> tuple[dict, dict]:
    """scripts/run.py into a fresh folder: (results, arrays)."""
    sys.path.insert(0, str(REPO / "src"))
    from mh_dec_mcts.experiment import load_arrays, load_results

    out = tmp / "run"
    sh(["scripts/run.py", config, *args, "--jobs", "8", "--out", str(out)])
    (folder,) = list(out.iterdir())
    res, arrays = load_results(folder), load_arrays(folder)
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()
    return res, arrays


def _stats(curves: list, key: str) -> dict:
    a = [c[key] for c in curves]
    return {"mean": np.mean(a, 0).tolist(), "std": np.std(a, 0).tolist()}


def run_single_case(tmp: Path, name, config, extra, runs, steps) -> dict:
    """Old run_single.py: curves.json (mean / std over runs per planner of entropy, MSE, coverage,
    altitude; coverage = Pierdicca's) and run0.npz (run-0 poses per planner, truth)."""
    extra = list(extra)
    planners = ["ig", "random", "sweep"]
    if "--planners" in extra:
        k = extra.index("--planners")
        planners, extra = [extra[k + 1]], extra[:k] + extra[k + 2:]
    res, arrays = run_py(tmp, config, ["--planners", *planners, *extra, "--runs", str(runs),
                                       "--steps", str(steps), "--save-poses"])
    names = {"entropy": "entropy", "mse": "mse", "coverage": "coverage_paper", "altitude": "alt"}
    curves, run0 = {}, {}
    for k, c in enumerate(res["conditions"]):
        rs = [r["curves"] for r in c["runs"]]
        curves[c["planner"]] = {m: _stats(rs, key) for m, key in names.items()}
        run0[f"poses_{c['planner']}"] = arrays[f"c{k}_r0_poses"][0].tolist()
    cfg = res["meta"]["config_resolved"]
    tag = cfg["name"] + (f"_r{cfg['field']['r']:g}" if "r" in cfg["field"] else "") + f"_{cfg['run']['weights']}"
    return strip({"folder": tag, "planners": planners, "curves": curves,
                  "run0": dict(sorted(run0.items())), "truth_sum": int(arrays["truth_r0"].sum())})


def run_multi_case(tmp: Path) -> dict:
    """Old run_multi.py: one JSON per condition, mean / std over runs of MSE, entropy, altitude
    (averaged over the UAVs) and misalignment."""
    config, planners, sharing, ranges, ns, runs, steps = MULTI_CASE
    res, _ = run_py(tmp, config, ["--planners", *planners, "--sharing", *sharing, "--ranges", *map(str, ranges),
                                  "--n", *map(str, ns), "--runs", str(runs), "--steps", str(steps)])
    names = {"mse": "mse", "entropy": "entropy", "altitude": "alt", "misalignment": "misalignment"}
    out = {}
    for c in res["conditions"]:
        rs = [r["curves"] for r in c["runs"]]
        out[f"{c['planner']}_{c['sharing']}_R{c['range']}_N{c['n']}"] = {m: _stats(rs, k) for m, k in names.items()}
    return strip(dict(sorted(out.items())))


MYOPIA_MEAN = ("mse", "team_entropy", "alt", "coverage", "plan", "overlap", "connected", "misalignment", "backfill")
OLD_SCALARS = ("avg_mse", "final_mse", "stall", "final_cov", "t80", "t95", "altitude", "balance")
ONE_RUN_CURVES = ("mse", "entropy", "team_entropy", "alt", "coverage", "misalignment", "overlap", "connected",
                  "backfill")


def as_runs_json(res: dict, n: int) -> dict:
    """results.json -> what the old myopia.py saved as runs.json."""
    m = res["meta"]
    saved = {"meta": {"config": m["config"], "setting": m["setting"], "runs": m["runs"], "steps": m["steps"],
                      "n": n, "trap": m["trap"]}, "planners": {}}
    for c in res["conditions"]:
        rs = [r["curves"] for r in c["runs"]]
        saved["planners"][c["planner"]] = {
            "runs": [{s: r["scalars"][s] for s in OLD_SCALARS} for r in c["runs"]],
            "mean": {key: np.mean([x[key] for x in rs], axis=0).tolist() for key in MYOPIA_MEAN},
            "std": {key: np.std([x[key] for x in rs], axis=0).tolist() for key in MYOPIA_MEAN}}
    return saved


def run_myopia_case(tmp: Path, name, config, setting, n, runs, steps) -> dict:
    """Old myopia.py: runs.json (per-run scalars, mean / std curves) and every per-run curve and
    scalar of one_run (with the poses and the first-seen map)."""
    res, arrays = run_py(tmp, config, ["--setting", setting, "--n", str(n), "--runs", str(runs),
                                       "--steps", str(steps), "--save-poses", "--planners", *MYOPIA_SPECS])
    one = {}
    for k, c in enumerate(res["conditions"]):
        for r in c["runs"]:
            i = r["run"]
            first = arrays[f"c{k}_r{i}_first"]
            one[f"{c['planner']}|{i}"] = {**{key: r["curves"][key] for key in ONE_RUN_CURVES},
                                         **{key: r["scalars"][key] for key in ("altitude", "balance", "stall")},
                                         "first_sum": int(first.astype(np.int64).sum()),
                                         "first_seen": int((first >= 0).sum()),
                                         "poses": arrays[f"c{k}_r{i}_poses"].tolist()}
    return strip({"runs_json": as_runs_json(res, n), "one_run": one})


def run_tune_case(tmp: Path) -> dict:
    config, param, runs, steps = TUNE_CASE
    out = tmp / "tune"
    sh(["scripts/tune_decmcts.py", config, "--param", *param, "--runs", str(runs), "--steps", str(steps),
        "--jobs", "8", "--out", str(out)])
    (folder,) = list(out.iterdir())
    return strip(json.loads((folder / "tune.json").read_text()))


def run_sweep_case(tmp: Path) -> dict:
    conditions, horizons, runs, steps = SWEEP_CASE
    work = tmp / "sweep"
    work.mkdir()
    for d in ("scripts", "configs"):  # the script writes results/ and paper/ relative to its cwd
        (work / d).symlink_to(REPO / d)
    sh([str(REPO / "scripts/horizon_sweep.py"), "--conditions", *conditions, "--horizons", *map(str, horizons),
        "--runs", str(runs), "--steps", str(steps), "--jobs", "8"], cwd=work)
    from mh_dec_mcts.experiment import load_results

    n = {"square_n1": 1, "square_n4": 4, "rect_n1": 1, "rect_n4": 4}
    raw = {c: as_runs_json(load_results(work / "results/horizon_sweep" / c), n[c]) for c in conditions}
    import csv

    with open(work / "paper/figures/horizon_sweep.csv") as f:
        rows = [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
    return strip({"runs_json": raw, "csv": rows})


# ------------------------------------------------------------------ all cases
def collect(tmp: Path, only: list[str] | None = None) -> dict:
    tmp = Path(tmp)
    jobs = {**{k: (run_single_case, (tmp, k, *v)) for k, v in SINGLE_CASES.items()},
            "multi_fig5": (run_multi_case, (tmp,)),
            **{k: (run_myopia_case, (tmp, k, *v)) for k, v in MYOPIA_CASES.items()},
            "tune": (run_tune_case, (tmp,)),
            "horizon_sweep": (run_sweep_case, (tmp,))}
    return {k: f(*a) for k, (f, a) in jobs.items() if only is None or k in only}


def diff(a, b, path="") -> list[str]:
    """Paths where a and b differ (exact equality; NaN equals NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing on one side" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b and type(a) is type(b) else [f"{path}: {a!r} != {b!r}"]
