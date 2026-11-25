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




def ms(mu_sd) -> dict:
    return {"mean": mu_sd[0].tolist(), "std": mu_sd[1].tolist()}


def run_single_case(tmp: Path, name, config, extra, runs, steps) -> dict:
    """The former single-UAV runner: curves.json (mean / std over runs per planner of entropy, MSE, coverage,
    altitude; coverage = Pierdicca's) and run0.npz (run-0 poses per planner, truth)."""
    extra = list(extra)
    planners = ["ig", "random", "sweep"]
    if "--planners" in extra:
        k = extra.index("--planners")
        planners, extra = [extra[k + 1]], extra[:k] + extra[k + 2:]
    res, arrays = run_py(tmp, config, ["--planners", *planners, *extra, "--runs", str(runs),
                                       "--steps", str(steps)])
    from mh_dec_mcts.plots import paper  # what the figures plot: *_arrays

    if name == "single_fig4":  # plotted by Fig. 4 (one result per weights and r)
        cfg = res["meta"]["config_resolved"]
        (data,) = paper.fig4_arrays({(cfg["run"]["weights"], cfg["field"]["r"]): res["conditions"][0]}).values()
        plotted = {"ig": data}
    else:
        plotted = paper.fig3_arrays(res)
    curves = {p: {m: ms(d[m]) for m in ("entropy", "mse", "coverage", "altitude")} for p, d in plotted.items()}
    run0 = {}
    for k, c in enumerate(res["conditions"]):
        run0[f"poses_{c['planner']}"] = arrays[f"c{k}_r0_poses"][0].tolist()
    cfg = res["meta"]["config_resolved"]
    tag = cfg["name"] + (f"_r{cfg['field']['r']:g}" if "r" in cfg["field"] else "") + f"_{cfg['run']['weights']}"
    return strip({"folder": tag, "planners": planners, "curves": curves,
                  "run0": dict(sorted(run0.items())), "truth_sum": int(arrays["truth_r0"].sum())})


def run_multi_case(tmp: Path) -> dict:
    """The former Fig. 5 runner: one JSON per condition, mean / std over runs of MSE, entropy, altitude
    (averaged over the UAVs) and misalignment."""
    config, planners, sharing, ranges, ns, runs, steps = MULTI_CASE
    res, _ = run_py(tmp, config, ["--planners", *planners, "--sharing", *sharing, "--ranges", *map(str, ranges),
                                  "--n", *map(str, ns), "--runs", str(runs), "--steps", str(steps)])
    from mh_dec_mcts.plots import paper
    from mh_dec_mcts.plots.common import band

    plotted = paper.fig5_arrays(res)  # mse, altitude, misalignment as Fig. 5 draws them
    out = {}
    for c in res["conditions"]:
        k = paper.fig5_key(c["planner"], c["sharing"], c["range"], c["n"])
        out[k] = {**{m: ms(v) for m, v in plotted[k].items()}, "entropy": ms(band(c, "entropy"))}
    return strip(dict(sorted(out.items())))


MYOPIA_MEAN = ("mse", "team_entropy", "alt", "coverage", "plan", "overlap", "connected", "misalignment", "backfill")
OLD_SCALARS = ("avg_mse", "final_mse", "stall", "final_cov", "t80", "t95", "altitude", "balance")
ONE_RUN_CURVES = ("mse", "entropy", "team_entropy", "alt", "coverage", "misalignment", "overlap", "connected",
                  "backfill")


def as_runs_json(res: dict, n: int) -> dict:
    """results.json -> what the former planner-comparison script saved as runs.json."""
    m = res["meta"]
    saved = {"meta": {"config": m["config"], "setting": m["setting"], "runs": m["runs"], "steps": m["steps"],
                      "n": n, "trap": m["trap"]}, "planners": {}}
    from mh_dec_mcts.plots import mission
    from mh_dec_mcts.plots.common import band

    plotted = mission.arrays(res["conditions"])  # what the mission figure draws, the rest as stored
    for c in res["conditions"]:
        bands = {key: plotted[c["planner"]].get(key) or band(c, key) for key in MYOPIA_MEAN}
        saved["planners"][c["planner"]] = {
            "runs": [{s: r["scalars"][s] for s in OLD_SCALARS} for r in c["runs"]],
            "mean": {key: b[0].tolist() for key, b in bands.items()},
            "std": {key: b[1].tolist() for key, b in bands.items()}}
    return saved


def run_myopia_case(tmp: Path, name, config, setting, n, runs, steps) -> dict:
    """The former planner-comparison script: runs.json (per-run scalars, mean / std curves) and every per-run curve and
    scalar of one_run (with the poses and the first-seen map)."""
    res, arrays = run_py(tmp, config, ["--setting", setting, "--n", str(n), "--runs", str(runs),
                                       "--steps", str(steps), "--planners", *MYOPIA_SPECS])
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


def run_tune(tmp: Path, args: list[str]) -> tuple[dict, dict, Path]:
    """scripts/tune.py into a fresh folder: (tune.json, results, folder)."""
    from mh_dec_mcts.experiment import load_results

    out = tmp / "tune"
    sh(["scripts/tune.py", *args, "--jobs", "8", "--out", str(out)])
    (folder,) = list(out.iterdir())
    return json.loads((folder / "tune.json").read_text()), load_results(folder), folder


def run_tune_case(tmp: Path) -> dict:
    """The former Dec-MCTS tuning script's tune.json: base + one row per value, vs greedy."""
    config, param, runs, steps = TUNE_CASE
    t, _, _ = run_tune(tmp, [config, "--param", *param, "--runs", str(runs), "--steps", str(steps)])
    keep = ("param", "value", "mcts", "d_auc", "se_auc", "d_final", "se_final")
    return strip({"setting": t["setting"], "seed": t["seed"], "runs": t["runs"], "steps": t["steps"],
                  "base": t["base"], "rows": [{k: r[k] for k in keep} for r in t["rows"] if r["param"] != "greedy"]})


def run_sweep_case(tmp: Path) -> dict:
    """The former horizon sweep (square, one UAV): its raw runs and summary CSV rows, now through
    tune.py (greedy rollouts, the config's seed); planner names in the sweep's spelling."""
    conditions, horizons, runs, steps = SWEEP_CASE
    assert conditions == ["square_n1"]
    t, res, _ = run_tune(tmp, ["configs/swarm.yaml", "--setting", "IGd_BM_R5", "--n", "1", "--base", "rollout=greedy",
                               "--param", "horizon", *map(str, horizons), "--runs", str(runs), "--steps", str(steps),
                               "--seed", "42"])
    old = {"greedy": "greedy", **{f"decmcts:rollout=greedy,horizon={h}": f"decmcts:horizon={h},rollout=greedy"
                                  for h in horizons}}
    res["conditions"] = [{**c, "planner": old[c["planner"]]} for c in res["conditions"] if c["planner"] in old]
    cols = ("runs", "avg_mse", "d_avg_pct", "d_avg_ci", "final_mse", "d_final_pct", "d_final_ci", "worst_avg_mse",
            "p90_avg_mse", "stalled")
    rows = [{"condition": "square_n1", "planner": old[r["spec"]], "horizon": str(r["value"] or 0),
             **{k: str(r[k]) for k in cols}} for r in t["rows"] if r["spec"] in old]
    return strip({"runs_json": {"square_n1": as_runs_json(res, 1)}, "csv": rows})


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
