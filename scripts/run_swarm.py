"""Legacy swarm comparison: every planner in the two settings of configs/swarm.yaml.

    python scripts/run_swarm.py configs/swarm.yaml
    python scripts/run_swarm.py configs/swarm.yaml --planners greedy --settings IGd_BM_R5 --runs 2 --steps 20
    python scripts/run_swarm.py configs/swarm.yaml --set horizon=6             # mcts override

Planner names: greedy (the setting's `greedy`: ig or igd), decmcts (plans exchanged with every
agent within the setting's comms range). Every planner sees the same fields, starts and
sensor noise per run (paired). Output in results/swarm/<name>/:
    <setting>__<planner>.npz   per run x step: entropy, mse, altitude (agent means), misalignment
    <setting>__<planner>.png   legacy viewer of run 0 (field, agent 0 final belief, all paths)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from mh_dec_mcts.planning.dec_mcts import make_dec_mcts
from mh_dec_mcts.scenario import load_config, make_scenario, make_streams, mapping_weights
from mh_dec_mcts.sim.multi import run_multi
from mh_dec_mcts.viewer import plot_terrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--planners", nargs="+")
    ap.add_argument("--settings", nargs="+")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--set", action="append", default=[], help="mcts override, e.g. horizon=6")
    ap.add_argument("--out", default="results/swarm")
    args = ap.parse_args()

    base = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        base["mcts"][k] = yaml.safe_load(v)
    runs, steps = args.runs or base["run"]["runs"], args.steps or base["run"]["steps"]
    out = Path(args.out) / base["name"]
    out.mkdir(parents=True, exist_ok=True)
    print("mcts:", base["mcts"], "| per-setting overrides:",
          {k: v["mcts"] for k, v in base["settings"].items() if "mcts" in v})

    for setting in args.settings or list(base["settings"]):
        s = base["settings"][setting]
        for planner in args.planners or base["planners"]:
            run_condition(base, setting, s, planner, runs, steps, out)


def run_condition(base, setting, s, planner, runs, steps, out):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    cfg["comms"] = {"range": s["range"], "sharing": s["sharing"]}
    cfg["mcts"] = {**base["mcts"], **s.get("mcts", {})}  # optional per-setting overrides
    if planner == "greedy":
        name = s["greedy"]
    elif planner == "decmcts":
        name = "decmcts"
    else:
        raise ValueError(f"unknown planner '{planner}' (mh is not ported yet)")
    tie_eps = 0.0 if cfg["compat"] == "pierdicca" else 1e-4
    streams = make_streams(cfg)  # None unless compat pierdicca; else seeding is per run
    curves = {"entropy": [], "mse": [], "altitude": [], "misalignment": []}
    t0 = time.time()
    for i in range(runs):
        sc = make_scenario(cfg, run=i, streams=streams)
        dec = make_dec_mcts(cfg, sc) if name == "decmcts" else None
        res = run_multi(sc, name, steps, mapping_weights(cfg), s["sharing"], s["range"], tie_eps, dec)
        for m in ("entropy", "mse", "altitude"):
            curves[m].append(np.mean(getattr(res, m), axis=1))
        curves["misalignment"].append(res.misalignment)
        if i == 0:
            plot_terrain(out / f"{setting}__{planner}.png", sc.lattice, sc.truth, res.beliefs[0], res.poses,
                         title=f"{planner} ({name}) | {setting} | run 0, step {steps}")
        print(f"  {setting} {planner}: run {i + 1}/{runs} ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out / f"{setting}__{planner}.npz", **{m: np.array(v) for m, v in curves.items()})
    mse = np.array(curves["mse"])[:, -1]
    print(f"{setting} {planner} ({name}): final MSE {mse.mean():.5f} +- {mse.std() / np.sqrt(runs):.5f}")


if __name__ == "__main__":
    main()
