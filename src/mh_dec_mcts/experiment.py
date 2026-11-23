"""Experiments: planner specs and one run of one planner on one scenario.

A planner spec is a planner name with optional overrides: "greedy" (the setting's IG / IGd),
"ig", "igd", "random", "sweep", or "decmcts" / "mhmcts" followed by ":key=value,...". Keys of
MHConfig (tile, g2_weight, ...) go to the config section `mh`, the others to `mcts`.
"""

import json
import time
from pathlib import Path

import numpy as np
import yaml

from mh_dec_mcts.comms.network import neighbours
from mh_dec_mcts.planning.mh_mcts import split_overrides
from mh_dec_mcts.planning.planners import make_planner
from mh_dec_mcts.scenario import make_scenario, mapping_weights
from mh_dec_mcts.sim.metrics import coverage
from mh_dec_mcts.sim.multi import run_multi


def parse_spec(spec: str) -> tuple[str, dict]:
    """'decmcts:horizon=8,rollout=greedy' -> ('decmcts', {'horizon': 8, 'rollout': 'greedy'})"""
    name, _, rest = spec.partition(":")
    return name, {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in rest.split(",") if kv)}


parse = parse_spec  # old name (scripts/myopia.py)


def longest_stall(h: np.ndarray, rate: float, left: float = 0.2) -> int:
    """Longest run of steps where entropy falls by < rate * (what is left) per step, while more
    than `left` of the initial entropy remains."""
    best = cur = 0
    for t in range(len(h) - 1):
        if h[t] > left * h[0] and (h[t] - h[t + 1]) < rate * h[t]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def paper_coverage(lat, shape: tuple[int, int], poses: list, steps: int) -> list[float]:
    """Coverage as in Pierdicca's figures (sim/single.py): at step t, the fraction of the field
    inside a footprint of any UAV at any of its poses 0..t (start included), recorded before the
    step's move. one_run's `coverage` instead counts the flown poses only, after the move."""
    seen = np.zeros(shape, dtype=bool)
    out = []
    for t in range(steps):
        for path in poses:
            i0, i1, j0, j1 = lat.footprint(path[t])
            seen[i0:i1, j0:j1] = True
        out.append(coverage(seen))
    return out


def one_run(cfg: dict, setting: dict, spec: str, run: int, steps: int, late: int,
            stall_rate: float = 0.01, streams: dict | None = None) -> dict:
    """One run of planner `spec` in a team setting {"sharing", "range", "greedy"}: per-step curves
    (averaged over the UAVs where per UAV), per-run scalars, the poses, the first-seen map and the
    ground truth. streams: the experiment's generators for compat pierdicca (make_streams), the
    runs of one experiment built in order 0, 1, 2, ... with the same object."""
    name, over = parse(spec)
    mcts_over, mh_over = split_overrides(over)  # e.g. mhmcts:horizon=4,tile=12.5
    cfg = {**cfg, "mcts": {**cfg["mcts"], **mcts_over}, "mh": {**cfg.get("mh", {}), **mh_over}}
    sc = make_scenario(cfg, run, streams)
    planner = setting["greedy"] if name == "greedy" else name
    pl = make_planner(planner, cfg, sc, setting["range"])
    t0, w0 = time.process_time(), time.perf_counter()
    res = run_multi(sc, pl, steps, mapping_weights(cfg), setting["sharing"], setting["range"])
    cpu = (time.process_time() - t0) / steps  # CPU seconds per step (whole team)
    wall = time.perf_counter() - w0  # wall seconds of the run: > CPU when processes share cores
    lat = sc.lattice
    first = np.full(sc.grid.shape, -1, dtype=np.int32)
    col_first = np.full(sc.grid.shape[1], -1)  # step at which any cell of each column was first seen
    cov, back, n_back = [], [], 0
    overlap, connected = [], []
    credit = np.zeros(len(res.poses))  # cells each UAV saw first
    count = np.zeros(sc.grid.shape, dtype=np.int16)
    for t in range(steps):
        fps = [lat.footprint(path[t + 1]) for path in res.poses]
        for a, (i0, i1, j0, j1) in enumerate(fps):
            win, cf = first[i0:i1, j0:j1], col_first[j0:j1]
            new = win < 0
            n_back += int(new[:, (cf >= 0) & (t - cf > late)].sum())  # columns flown over long ago
            credit[a] += new.sum()
            win[new] = t
            cf[(cf < 0) & new.any(axis=0)] = t
            count[i0:i1, j0:j1] += 1
        i0, i1 = min(f[0] for f in fps), max(f[1] for f in fps)
        j0, j1 = min(f[2] for f in fps), max(f[3] for f in fps)
        box_ = count[i0:i1, j0:j1]
        overlap.append(float((box_ > 1).sum() / max(1, (box_ > 0).sum())))
        count[i0:i1, j0:j1] = 0
        nb = neighbours([path[t + 1] for path in res.poses], setting["range"])
        connected.append(float(np.mean([len(x) > 0 for x in nb])))
        cov.append(float((first >= 0).mean()))
        back.append(n_back / first.size)
    team_mse = np.mean(res.mse, axis=1)
    team_h = np.mean(res.entropy, axis=1)
    return {"coverage": cov, "backfill": back, "mse": team_mse.tolist(),
            "entropy": team_h.tolist(), "team_entropy": list(res.team_entropy),
            "plan": list(res.plan_time), "plan_per_step": float(np.mean(res.plan_time)),
            "alt": np.mean(res.altitude, axis=1).tolist(), "altitude": float(np.mean(res.altitude)),
            "overlap": overlap, "connected": connected, "misalignment": list(res.misalignment),
            "balance": float(credit.std() / max(credit.mean(), 1e-12)),
            "stall": longest_stall(team_h, stall_rate), "first": first, "poses": res.poses, "cpu": cpu,
            "wall": wall, "coverage_paper": paper_coverage(lat, sc.grid.shape, res.poses, steps),
            "truth": sc.truth}


# ------------------------------------------------------------------ result format (docs/results_format.md)
SCHEMA = 1
CURVES = ("mse", "entropy", "team_entropy", "alt", "coverage", "coverage_paper", "plan", "misalignment",
          "overlap", "connected", "backfill")  # per-step curves of a run, as one_run returns them


def condition_id(spec: str, n: int, sharing: str | None, range_: int | None, setting: str | None) -> str:
    return f"{spec}|n={n}|sharing={sharing}|range={range_}|setting={setting}"


def run_record(r: dict, run: int, run_seed: int) -> dict:
    """One run of one_run in the result format: per-step curves and per-run scalars. The scalars
    keep the names (and the computation) of the old myopia runs.json."""
    cov = r["coverage"]
    return {"run": run, "run_seed": run_seed,
            "curves": {k: r[k] for k in CURVES if k in r},
            "scalars": {"avg_mse": float(np.mean(r["mse"])), "final_mse": float(r["mse"][-1]),
                        "stall": int(r["stall"]), "final_cov": float(cov[-1]),
                        "t80": next((t for t, c in enumerate(cov) if c >= 0.8), None),
                        "t95": next((t for t, c in enumerate(cov) if c >= 0.95), None),
                        "altitude": r["altitude"], "balance": r["balance"], "cpu_per_step": r["cpu"],
                        "plan_per_step": r["plan_per_step"], "wall": r["wall"]}}


def make_meta(config: str, config_resolved: dict, argv: list[str]) -> dict:
    import datetime
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"config": config, "config_resolved": config_resolved, "git_commit": commit,
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "command": list(argv)}


def save_results(folder, meta: dict, conditions: list[dict], arrays: dict | None = None) -> Path:
    """results.json (+ results.npz with `arrays`, e.g. poses and first-seen maps) in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps({"schema": SCHEMA, "meta": meta, "conditions": conditions}))
    if arrays:
        np.savez_compressed(folder / "results.npz", **arrays)
    return folder / "results.json"


def result_file(folder) -> Path | None:
    """The results file of a folder: results.json, else an old runs.json (or a single renamed
    runs*.json, e.g. runs_rect_n1.json), else None."""
    folder = Path(folder)
    if (folder / "results.json").exists():
        return folder / "results.json"
    if (folder / "runs.json").exists():
        return folder / "runs.json"
    old = sorted(folder.glob("runs*.json"))
    return old[0] if len(old) == 1 else None


def load_results(path) -> dict:
    """A results.json (or the folder holding it) in the result format. Also reads the old
    myopia / horizon-sweep runs.json: per-run scalars and mean (and std) curves, no per-run
    curves; returned in the same shape with curves = None and the old curves as mean_curves /
    std_curves of each condition."""
    path = Path(path)
    if path.is_dir():
        path = result_file(path)
        if path is None:
            raise FileNotFoundError(f"no results.json or runs*.json in {path}")
    d = json.loads(path.read_text())
    if "schema" in d:
        return d
    m = d["meta"]
    conds = []
    for spec, p in d["planners"].items():
        conds.append({"id": condition_id(spec, m.get("n"), None, None, m.get("setting")), "planner": spec,
                      "n": m.get("n"), "sharing": None, "range": None, "setting": m.get("setting"),
                      "steps": m.get("steps"),
                      "runs": [{"run": i, "run_seed": None, "curves": None, "scalars": s}
                               for i, s in enumerate(p["runs"])],
                      "mean_curves": p.get("mean"), "std_curves": p.get("std")})
    meta = {"config": m.get("config"), "config_resolved": None, "git_commit": None, "created": None,
            "command": None, "old_format": str(path), **{k: v for k, v in m.items() if k != "config"}}
    return {"schema": 0, "meta": meta, "conditions": conds}


def load_arrays(path) -> dict:
    """results.npz next to a results.json (poses, first-seen maps, truth), {} if absent."""
    path = Path(path)
    f = (path if path.is_dir() else path.parent) / "results.npz"
    if not f.exists():
        return {}
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ statistics over runs
def curve_stats(curves: list) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std (ddof 0, as Pierdicca) over runs of per-step curves (runs x steps)."""
    return np.mean(curves, 0), np.std(curves, 0)


def se(x) -> float:
    """Standard error over runs: std / sqrt(runs), NaN-aware (runs that never reach a level)."""
    return np.nanstd(x) / np.sqrt(len(x))


def paired_diff(x, ref) -> np.ndarray:
    """Per-run differences against a reference planner on the same runs (paired design)."""
    return np.asarray(x) - np.asarray(ref)
