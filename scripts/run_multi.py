"""Multi-agent experiments (paper Fig. 5). One JSON of curves per condition.

    python scripts/run_multi.py configs/pierdicca_fig5.yaml                      # full Fig. 5 grid
    python scripts/run_multi.py configs/pierdicca_fig5.yaml --planners ig --sharing BM --ranges 16 --n 4
    python scripts/run_multi.py configs/pierdicca_fig5.yaml --runs 2 --steps 20  # quick
    python scripts/run_multi.py configs/decmcts.yaml --planners decmcts igd --sharing none

Output: results/multi/<name>/<planner>_<sharing>_R<range>_N<n>.json with mean/std over runs of
the per-agent averaged MSE, entropy, altitude, and the misalignment.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mh_dec_mcts.planning.dec_mcts import make_dec_mcts
from mh_dec_mcts.scenario import load_config, make_scenario, make_streams, mapping_weights
from mh_dec_mcts.sim.multi import run_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--planners", nargs="+", default=["ig", "igd"], choices=["ig", "igd", "decmcts"])
    ap.add_argument("--sharing", nargs="+", default=["none", "BS", "BM"], choices=["none", "BS", "BM"])
    ap.add_argument("--ranges", nargs="+", type=int, default=[5, 16])
    ap.add_argument("--n", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--runs", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--out", default="results/multi")
    args = ap.parse_args()

    base = load_config(args.config)
    out = Path(args.out) / base["name"]
    out.mkdir(parents=True, exist_ok=True)
    for planner in args.planners:
        for sharing in args.sharing:
            for range_ in args.ranges:
                for n in args.n:
                    if n == 1 and (sharing != "none" or range_ != args.ranges[0]):
                        continue  # one agent: nothing to share, range irrelevant
                    run_condition(args, base, planner, sharing, range_, n, out)


def run_condition(args, base, planner, sharing, range_, n, out):
    over = {"agents": {"n": n}, "comms": {"range": range_, "sharing": sharing}, "run": {"planner": planner}}
    if args.runs:
        over["run"]["runs"] = args.runs
    if args.steps:
        over["run"]["steps"] = args.steps
    cfg = load_config_dict(base, over)
    rc = cfg["run"]
    tie_eps = 0.0 if cfg["compat"] == "pierdicca" else 1e-4
    streams = make_streams(cfg)  # one experiment = fresh generators, as in his simulate.py
    curves = {"mse": [], "entropy": [], "altitude": [], "misalignment": []}
    tag = f"{planner}_{sharing}_R{range_}_N{n}"
    t0 = time.time()
    for i in range(rc["runs"]):
        t_run = time.time()
        sc = make_scenario(cfg, run=i, streams=streams)
        dec = make_dec_mcts(cfg, sc) if planner == "decmcts" else None
        res = run_multi(sc, planner, rc["steps"], mapping_weights(cfg), sharing, range_, tie_eps, dec)
        for m in ("mse", "entropy", "altitude"):
            curves[m].append(np.mean(getattr(res, m), axis=1))  # average over agents
        curves["misalignment"].append(res.misalignment)
        print(f"  {tag}: run {i + 1}/{rc['runs']} done in {time.time() - t_run:.1f} s "
              f"(total {time.time() - t0:.0f} s)", flush=True)
    summary = {m: {"mean": np.mean(v, 0).tolist(), "std": np.std(v, 0).tolist()} for m, v in curves.items()}
    (out / f"{tag}.json").write_text(json.dumps({"planner": planner, "sharing": sharing, "range": range_,
                                                 "n": n, "config": cfg, "curves": summary}))
    print(f"{tag}: final MSE {summary['mse']['mean'][-1]:.5f}, "
          f"peak misalignment {max(summary['misalignment']['mean']):.4f}")


def load_config_dict(base: dict, over: dict) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for section, values in over.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


if __name__ == "__main__":
    main()
