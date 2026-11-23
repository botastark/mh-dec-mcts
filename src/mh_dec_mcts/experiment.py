"""Experiments: planner specs and one run of one planner on one scenario.

A planner spec is a planner name with optional overrides: "greedy" (the setting's IG / IGd),
"ig", "igd", "random", "sweep", or "decmcts" / "mhmcts" followed by ":key=value,...". Keys of
MHConfig (tile, g2_weight, ...) go to the config section `mh`, the others to `mcts`.
"""

import time

import numpy as np
import yaml

from mh_dec_mcts.comms.network import neighbours
from mh_dec_mcts.planning.mh_mcts import split_overrides
from mh_dec_mcts.planning.planners import make_planner
from mh_dec_mcts.scenario import make_scenario, mapping_weights
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


def one_run(cfg: dict, setting: dict, spec: str, run: int, steps: int, late: int,
            stall_rate: float = 0.01) -> dict:
    name, over = parse(spec)
    mcts_over, mh_over = split_overrides(over)  # e.g. mhmcts:horizon=4,tile=12.5
    cfg = {**cfg, "mcts": {**cfg["mcts"], **mcts_over}, "mh": {**cfg.get("mh", {}), **mh_over}}
    sc = make_scenario(cfg, run)
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
            "wall": wall}
