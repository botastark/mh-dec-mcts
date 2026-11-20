"""Tune Dec-MCTS: one-at-a-time sweep around the config's `mcts`, scored against greedy, paired.

Each candidate changes ONE parameter of the base `mcts` section; every candidate and the greedy
baseline see the same fields / starts / noise (paired). Tune on other seeds than the ones you report
(default --seed 1000, the legacy sweep seed; report with swarm.yaml's seed 42).

Score per run = MSE averaged over all steps (area under the MSE curve: rewards mapping fast, not
only the end). Reported: candidate - greedy, mean +- s.e. over runs (negative = better than greedy).

    python scripts/tune_decmcts.py configs/swarm.yaml --setting IGd_BM_R5 --jobs 8 \\
        --param gamma 0.95 0.99 0.999 0.9999 --param cp 0.25 0.5 1.0 --param horizon 2 3 4 6
    python scripts/tune_decmcts.py configs/swarm.yaml --param variant distribution single --runs 4 --steps 30

Output: results/tune/<name>_<setting>/tune.json and tune.png (score per value, one panel per
parameter, dashed line = base config, 0 = greedy).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker (before numpy is imported)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from mh_dec_mcts.planning.dec_mcts import make_dec_mcts
from mh_dec_mcts.scenario import load_config, make_scenario, mapping_weights
from mh_dec_mcts.sim.multi import run_multi


def one_run(cfg: dict, setting: dict, planner: str, run: int, steps: int):
    """(mean MSE over steps, final MSE, seconds per step) of one run."""
    sc = make_scenario(cfg, run)
    dec = make_dec_mcts(cfg, sc) if planner == "decmcts" else None
    t = time.time()
    res = run_multi(sc, planner, steps, mapping_weights(cfg), setting["sharing"], setting["range"], 1e-4, dec)
    mse = np.mean(res.mse, axis=1)
    return float(mse.mean()), float(mse[-1]), (time.time() - t) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--setting", default="IGd_BM_R5")
    ap.add_argument("--param", nargs="+", action="append", default=[], metavar=("NAME", "VALUES"),
                    help="parameter name then values, e.g. --param gamma 0.99 0.999")
    ap.add_argument("--runs", type=int, help="default: the config's run.runs")
    ap.add_argument("--steps", type=int, help="default: the config's run.steps")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/tune")
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    args.steps = args.steps or cfg["run"]["steps"]
    args.runs = args.runs or cfg["run"]["runs"]
    if cfg["compat"] is not None:
        raise SystemExit("tune with compat: null (paired per-run seeding)")
    setting = cfg["settings"][args.setting]
    cfg["comms"] = {"range": setting["range"], "sharing": setting["sharing"]}
    base = dict(cfg["mcts"])
    cands = [("base", None, base)]
    for name, *values in args.param:
        for v in values:
            v = yaml.safe_load(v)
            cands.append((name, v, {**base, name: v}))
    print(f"{args.setting}, seed {args.seed}, {args.runs} runs x {args.steps} steps, base mcts {base}")

    jobs = [("greedy", None, None, i) for i in range(args.runs)]
    jobs += [(n, v, m, i) for n, v, m in cands for i in range(args.runs)]
    print(f"{len(jobs)} runs on {args.jobs} processes; progress every finished run", flush=True)
    t0, out = time.time(), [None] * len(jobs)
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {}
        for k, (n, v, m, i) in enumerate(jobs):
            c = {**cfg, "mcts": m} if m is not None else cfg
            futs[ex.submit(one_run, c, setting, setting["greedy"] if n == "greedy" else "decmcts", i, args.steps)] = k
        for done, f in enumerate(as_completed(futs), 1):
            k = futs[f]
            out[k] = f.result()
            n, v, _, i = jobs[k]
            el = time.time() - t0
            label = f"greedy ({setting['greedy']})" if n == "greedy" else "base" if n == "base" else f"{n}={v}"
            print(f"  [{done}/{len(jobs)}] {label} run {i}: {out[k][2]:.2f} s/step | elapsed {el / 60:.1f} min,"
                  f" ~{el / done * (len(jobs) - done) / 60:.0f} min left", flush=True)
    res = {}
    for (n, v, _, i), r in zip(jobs, out):
        res.setdefault((n, json.dumps(v)), []).append(r)

    g = np.array(res[("greedy", "null")])
    rows = []
    for n, v, m in cands:
        a = np.array(res[(n, json.dumps(v))])
        d_auc, d_fin = a[:, 0] - g[:, 0], a[:, 1] - g[:, 1]
        se = lambda x: float(x.std() / np.sqrt(len(x)))
        rows.append({"param": n, "value": v, "mcts": m, "d_auc": float(d_auc.mean()), "se_auc": se(d_auc),
                     "d_final": float(d_fin.mean()), "se_final": se(d_fin), "s_per_step": float(a[:, 2].mean())})
    print(f"greedy ({setting['greedy']}): mean MSE {g[:, 0].mean():.5f}, final {g[:, 1].mean():.5f}, "
          f"{g[:, 2].mean():.3f} s/step")
    print(f"{'param':>16s} {'value':>12s} {'dMSE(avg) vs greedy':>22s} {'(%)':>6s} {'dMSE(final)':>22s}"
          f" {'s/step':>7s}")
    for r in sorted(rows, key=lambda r: r["d_auc"]):
        print(f"{r['param']:>16s} {'(config)' if r['value'] is None else str(r['value']):>12s}"
              f" {r['d_auc']:+.5f} +- {r['se_auc']:.5f} {100 * r['d_auc'] / g[:, 0].mean():+5.1f}%"
              f" {r['d_final']:+.2e} +- {r['se_final']:.1e} {r['s_per_step']:7.2f}")

    folder = Path(args.out) / f"{cfg['name']}_{args.setting}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tune.json").write_text(json.dumps({"setting": args.setting, "seed": args.seed, "runs": args.runs,
                                                  "steps": args.steps, "base": base, "rows": rows}, indent=1))
    plot(rows, folder / "tune.png", f"{args.setting}: Dec-MCTS - greedy, MSE averaged over steps")
    print(f"-> {folder}")


def plot(rows, path, title):
    import matplotlib.pyplot as plt

    base = next(r for r in rows if r["param"] == "base")
    params = [p for p in dict.fromkeys(r["param"] for r in rows) if p != "base"]
    if not params:
        return
    fig, axs = plt.subplots(1, len(params), figsize=(4 * len(params), 3.5), squeeze=False)
    for ax, p in zip(axs[0], params):
        rs = [r for r in rows if r["param"] == p]
        x = np.arange(len(rs))
        ax.errorbar(x, [r["d_auc"] for r in rs], yerr=[2 * r["se_auc"] for r in rs], fmt="o", capsize=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(base["d_auc"], color="tab:gray", ls="--", lw=0.8)
        ax.set_xticks(x, [str(r["value"]) for r in rs])
        ax.set(title=p, xlabel=p)
        ax.grid(alpha=0.4)
    axs[0, 0].set_ylabel("dMSE vs greedy (+-2 s.e.)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
