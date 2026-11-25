"""Tune Dec-MCTS or MH-MCTS: one parameter at a time around a base, against greedy, paired runs.

    # Dec-MCTS around the config's `mcts` (seed 1000: tune on other runs than you report)
    python scripts/tune.py configs/swarm.yaml --setting IGd_BM_R5 --jobs 16 \\
        --param gamma 0.95 0.99 0.999 --param cp 0.25 0.5 1.0 --param horizon 2 3 4 6
    # MH-MCTS: keys of MHConfig (tile, g2_weight, hl_*) go to `mh`, the others to `mcts`
    python scripts/tune.py configs/swarm_rect.yaml --planner mhmcts --base horizon=4,rollout=greedy \\
        --n 1 --steps 300 --param tile 6.25 12.5 25 --param hl_rollouts 0 10
    # horizon sweep (greedy rollouts, the seeds you report), one call per field / team size
    python scripts/tune.py configs/swarm_rect.yaml --base rollout=greedy --param horizon 2 4 8 16 24 \\
        --n 1 --steps 300 --seed 42 --tag rect_n1
    # figure (any time later, several tunings side by side, e.g. the horizon sweep's conditions)
    python scripts/plot.py tune results/tune/square_n1 results/tune/square_n4 ...

Candidates: the base (--planner with --base overrides on top of the config) and, for every
--param value, the base with that one value changed. All of them and the greedy baseline (the
setting's IG / IGd) run through scripts/run.py on the same runs: same fields, starts and sensor
noise (paired). Score per run: MSE averaged over the steps (area under the MSE curve) and final
MSE, each as the difference to greedy on the same run.

Output in <out>/<tag>/ (tag default: <config name>_<setting>_<planner>), data only: run.py's
results.json / results.npz / results.txt, and
    tune.json   per team size and candidate: the spec, its resolved `mcts` (and `mh`) section,
                difference to greedy of avg / final MSE (mean +- s.e., and % of greedy's mean
                +- 95 % CI, paired), worst and 90th-percentile avg MSE, share of stalled runs,
                CPU and wall seconds per step
    tune.csv    the same rows
Figure: scripts/plot.py tune <folder> [...]: columns = (team size, parameter), rows = avg and final
MSE vs greedy (%, 95 % CI), stalled runs, CPU s per step; dashed grey = base, black = greedy.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one maths thread per process, before numpy is imported

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from run import main as run_py

from mh_dec_mcts.experiment import load_results, parse_spec
from mh_dec_mcts.planning.mh_mcts import split_overrides
from mh_dec_mcts.scenario import load_config

FIELDS = ("n", "param", "value", "spec", "runs", "avg_mse", "d_auc", "se_auc", "d_avg_pct", "d_avg_ci",
          "final_mse", "d_final", "se_final", "d_final_pct", "d_final_ci", "worst_avg_mse", "p90_avg_mse",
          "stalled", "cpu_per_step", "s_per_step")


def spec_of(planner: str, items: list[tuple[str, str]]) -> str:
    return planner + (":" + ",".join(f"{k}={v}" for k, v in items) if items else "")


def candidates(planner: str, base: str, params: list[list[str]]) -> list[tuple[str, object, str]]:
    """(param, value, spec): the base ("base", None) first, then one change per value."""
    items = [tuple(kv.split("=", 1)) for kv in base.split(",") if kv] if base else []
    out = [("base", None, spec_of(planner, items))]
    for name, *values in params:
        for v in values:
            out.append((name, yaml.safe_load(v), spec_of(planner, [(k, x) for k, x in items if k != name] + [(name, v)])))
    return out


def sections(cfg: dict, spec: str) -> tuple[dict, dict]:
    """The `mcts` and `mh` sections a spec runs with (config + its overrides)."""
    mcts_over, mh_over = split_overrides(parse_spec(spec)[1])
    return {**cfg["mcts"], **mcts_over}, {**cfg.get("mh", {}), **mh_over}


def summarise(res: dict, cands: list, planner: str) -> list[dict]:
    """Per team size: greedy and every candidate, scored against greedy on the same runs."""
    cfg, trap = res["meta"]["config_resolved"], res["meta"]["trap"]
    rows = []
    for n in dict.fromkeys(c["n"] for c in res["conditions"]):
        by_spec = {c["planner"]: [r["scalars"] for r in c["runs"]] for c in res["conditions"] if c["n"] == n}
        steps = next(c["steps"] for c in res["conditions"] if c["n"] == n)
        g = by_spec["greedy"]
        g_avg, g_fin = np.array([r["avg_mse"] for r in g]), np.array([r["final_mse"] for r in g])
        for param, value, spec in [("greedy", None, "greedy"), *cands]:
            rs = by_spec[spec]
            avg, fin = np.array([r["avg_mse"] for r in rs]), np.array([r["final_mse"] for r in rs])
            d_auc, d_fin = avg - g_avg, fin - g_fin  # paired, per run
            pa, pf = 100 * d_auc / g_avg.mean(), 100 * d_fin / g_fin.mean()  # % of greedy's mean
            se = lambda x: float(x.std() / np.sqrt(len(x)))
            ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            row = {"n": n, "param": param, "value": value, "spec": spec, "runs": len(rs),
                   "avg_mse": avg.mean(), "d_auc": float(d_auc.mean()), "se_auc": se(d_auc),
                   "d_avg_pct": pa.mean(), "d_avg_ci": ci(pa),
                   "final_mse": fin.mean(), "d_final": float(d_fin.mean()), "se_final": se(d_fin),
                   "d_final_pct": pf.mean(), "d_final_ci": ci(pf),
                   "worst_avg_mse": avg.max(), "p90_avg_mse": np.percentile(avg, 90),
                   "stalled": sum(r["stall"] >= trap for r in rs) / len(rs),
                   "cpu_per_step": np.mean([r["cpu_per_step"] for r in rs]),
                   "s_per_step": float(np.mean([r["wall"] / steps for r in rs]))}
            if spec != "greedy":
                row["mcts"], mh = sections(cfg, spec)
                if planner == "mhmcts":
                    row["mh"] = mh
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--planner", default="decmcts", choices=["decmcts", "mhmcts"])
    ap.add_argument("--base", default="", help="overrides of the base candidate, e.g. horizon=4,rollout=greedy")
    ap.add_argument("--param", nargs="+", action="append", default=[], metavar=("NAME", "VALUES"),
                    help="parameter name then values, e.g. --param gamma 0.99 0.999")
    ap.add_argument("--setting", default="IGd_BM_R5")
    ap.add_argument("--n", nargs="+", type=int, help="team sizes (default: the config's)")
    ap.add_argument("--runs", type=int, help="default: the config's run.runs")
    ap.add_argument("--steps", type=int, help="default: the config's run.steps")
    ap.add_argument("--seed", type=int, default=1000, help="default 1000: tune on other runs than you report")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/tune")
    ap.add_argument("--tag", help="default: <config name>_<setting>_<planner>")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg["compat"] is not None:
        raise SystemExit("tune with compat: null (paired per-run seeding)")
    tag = args.tag or f"{cfg['name']}_{args.setting}_{args.planner}"
    cands = candidates(args.planner, args.base, args.param)
    specs = list(dict.fromkeys(["greedy", *(s for _, _, s in cands)]))
    argv_run = [args.config, "--setting", args.setting, "--seed", str(args.seed), "--jobs", str(args.jobs),
                "--out", args.out, "--tag", tag, "--planners", *specs]
    for opt in ("n", "runs", "steps"):
        v = getattr(args, opt)
        if v:
            argv_run += [f"--{opt}", *map(str, v if isinstance(v, list) else [v])]
    folder = run_py(argv_run)
    res = load_results(folder)
    rows = summarise(res, cands, args.planner)
    base_mcts, base_mh = sections(res["meta"]["config_resolved"], cands[0][2])
    m = res["meta"]
    out = {"tag": tag, "config": args.config, "planner": args.planner, "setting": args.setting, "seed": args.seed,
           "runs": m["runs"], "steps": m["steps"], "base": base_mcts, "rows": rows}
    if args.planner == "mhmcts":
        out["base_mh"] = base_mh
    (folder / "tune.json").write_text(json.dumps(out, indent=1, default=float))
    with open(folder / "tune.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'n':>2s} {'candidate':>44s} {'avg MSE vs greedy':>24s} {'final MSE vs greedy':>24s}"
          f" {'stalled':>8s} {'CPU s/step':>10s}")
    for r in rows:
        print(f"{r['n']:2d} {r['spec']:>44s} {r['d_avg_pct']:+8.1f}% +- {r['d_avg_ci']:5.1f}"
              f"      {r['d_final_pct']:+8.1f}% +- {r['d_final_ci']:5.1f}      {r['stalled']:6.2f} {r['cpu_per_step']:10.3f}")
    print(f"-> {folder}   (figure: python scripts/plot.py tune {folder})")
    return folder


if __name__ == "__main__":
    main()
