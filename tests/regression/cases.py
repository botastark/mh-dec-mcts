"""Regression cases for the runner refactor: small versions of every experiment the scripts run.

collect(tmp) runs every case through the current code and returns {case name: canonical data},
plain JSON-able values (floats kept exactly, never rounded). make_baseline.py stores that once;
test_baseline.py recomputes it and compares by exact equality. As scripts are replaced, only the
runners below change: the canonical data of each case must stay the same.

Timings (CPU / wall seconds, planning time) differ from run to run and are left out.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
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


def poses(path) -> list:
    return [[p.ix, p.iy, p.k] for p in path]


def sh(args, cwd=REPO):
    subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, check=True, stdout=subprocess.DEVNULL)


# ------------------------------------------------------------------ runners (today: the old scripts)
def run_single_case(tmp: Path, name, config, extra, runs, steps) -> dict:
    out = tmp / "single" / name
    sh(["scripts/run_single.py", config, *extra, "--runs", str(runs), "--steps", str(steps), "--out", str(out)])
    (folder,) = [d for d in out.iterdir() if d.is_dir()]
    curves = json.loads((folder / "curves.json").read_text())
    run0 = np.load(folder / "run0.npz")
    return strip({"folder": folder.name, "planners": curves["planners"], "curves": curves["curves"],
                  "run0": {k: run0[k].tolist() for k in sorted(run0.files) if k != "truth"},
                  "truth_sum": int(run0["truth"].sum())})


def run_multi_case(tmp: Path) -> dict:
    config, planners, sharing, ranges, ns, runs, steps = MULTI_CASE
    out = tmp / "multi"
    sh(["scripts/run_multi.py", config, "--planners", *planners, "--sharing", *sharing,
        "--ranges", *map(str, ranges), "--n", *map(str, ns), "--runs", str(runs), "--steps", str(steps),
        "--out", str(out)])
    (folder,) = list(out.iterdir())
    return strip({f.stem: json.loads(f.read_text())["curves"] for f in sorted(folder.glob("*.json"))})


def run_myopia_case(tmp: Path, name, config, setting, n, runs, steps) -> dict:
    out = tmp / "myopia"
    sh(["scripts/myopia.py", config, "--setting", setting, "--n", str(n), "--runs", str(runs),
        "--steps", str(steps), "--jobs", "8", "--out", str(out), "--tag", name, "--planners", *MYOPIA_SPECS])
    saved = json.loads((out / name / "runs.json").read_text())
    return strip({"runs_json": saved, "one_run": myopia_one_runs(config, setting, n, runs, steps)})


def _one_run_job(config, setting_name, n, spec, i, steps):
    from mh_dec_mcts.scenario import load_config

    sys.path.insert(0, str(REPO / "scripts"))
    from myopia import one_run  # T1 moves it to mh_dec_mcts.experiment

    cfg = load_config(str(REPO / config), agents={"n": n})
    setting = cfg["settings"][setting_name]
    cfg["comms"] = {"range": setting["range"], "sharing": setting["sharing"]}
    r = one_run(cfg, setting, spec, i, steps, 10, 0.01)
    first = r.pop("first")
    r["first_sum"] = int(first.astype(np.int64).sum())
    r["first_seen"] = int((first >= 0).sum())
    r["poses"] = [poses(p) for p in r["poses"]]
    return strip(r)


def myopia_one_runs(config, setting, n, runs, steps) -> dict:
    """Every per-run curve and scalar of one_run (runs.json keeps only scalars and means)."""
    with ProcessPoolExecutor(8) as ex:
        futs = {f"{s}|{i}": ex.submit(_one_run_job, config, setting, n, s, i, steps)
                for s in MYOPIA_SPECS for i in range(runs)}
        return {k: f.result() for k, f in futs.items()}


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
    raw = {c: json.loads((work / "results/horizon_sweep" / c / "runs.json").read_text()) for c in conditions}
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
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return []
    return [] if a == b and type(a) is type(b) else [f"{path}: {a!r} != {b!r}"]
