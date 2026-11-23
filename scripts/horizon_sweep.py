"""Horizon sweep: is there one good Dec-MCTS horizon? (motivation for multi-horizon planning)

Conditions: square 50 x 50 m vs rectangle 100 x 50 m, 1 vs 4 UAVs (setting IGd_BM_R5; with one
UAV the comms setting does not matter). Planners: greedy and Dec-MCTS with horizon 2, 4, 8, 16, 24
(greedy rollouts, everything else from the config's `mcts`). 20 paired runs each (config).

    python scripts/horizon_sweep.py --jobs 8                 # runs what is missing, then the figures
    python scripts/horizon_sweep.py --plot-only              # figures from existing results
    python scripts/horizon_sweep.py --conditions rect_n1 --horizons 2 8 16 --jobs 8
    python scripts/horizon_sweep.py --extra                  # + rectangle, 4 UAVs, IG_BS_Rinf
    python scripts/horizon_sweep.py --runs 2 --steps 10 --horizons 2 4 --jobs 2   # smoke test
        (then delete results/horizon_sweep/ before the real run, or pass --force)

Raw results: results/horizon_sweep/<condition>/results.json (older sweeps: runs.json, also read).
Paper figures: paper/figures/
    horizon_sweep.pdf / .png   per condition (columns) vs horizon: avg MSE and final MSE relative
                               to greedy (paired, mean +- 95 % CI), stalled runs, CPU per step;
                               greedy = dashed line
    horizon_sweep.csv          all numbers behind the figure
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from mh_dec_mcts.experiment import load_results, result_file

CONDITIONS = {  # name: (config, setting, UAVs, steps, label)
    "square_n1": ("configs/swarm.yaml", "IGd_BM_R5", 1, 300, "square 50x50 m, 1 UAV"),
    "square_n4": ("configs/swarm.yaml", "IGd_BM_R5", 4, 100, "square 50x50 m, 4 UAVs"),
    "rect_n1": ("configs/swarm_rect.yaml", "IGd_BM_R5", 1, 300, "rectangle 100x50 m, 1 UAV"),
    "rect_n4": ("configs/swarm_rect.yaml", "IGd_BM_R5", 4, 100, "rectangle 100x50 m, 4 UAVs"),
}
EXTRA = {"rect_n4_bs": ("configs/swarm_rect.yaml", "IG_BS_Rinf", 4, 100, "rectangle, 4 UAVs, full sharing")}
RAW = Path("results/horizon_sweep")
FIGS = Path("paper/figures")


def done(name: str) -> bool:
    return result_file(RAW / name) is not None


def spec(h: int) -> str:
    return f"decmcts:horizon={h},rollout=greedy"


def run(conditions, horizons, jobs, runs, steps_override, force):
    for name, (config, setting, n, steps, _) in conditions.items():
        if done(name) and not force:
            print(f"{name}: already done (--force to rerun)")
            continue
        cmd = [sys.executable, "scripts/myopia.py", config, "--setting", setting, "--n", str(n),
               "--steps", str(steps_override or steps), "--jobs", str(jobs), "--out", str(RAW), "--tag", name,
               "--planners", "greedy", *[spec(h) for h in horizons]]
        if runs:
            cmd += ["--runs", str(runs)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def summarise(conditions, horizons):
    """rows of: condition, planner, horizon, metric means and 95 % CIs (paired vs greedy)."""
    rows = []
    for name in conditions:
        if not done(name):
            print(f"{name}: no results yet, skipped")
            continue
        d = load_results(RAW / name)  # results.json, or the runs.json of older sweeps
        planners = {c["planner"]: [r["scalars"] for r in c["runs"]] for c in d["conditions"]}
        g = planners["greedy"]
        g_avg = np.array([r["avg_mse"] for r in g])
        g_fin = np.array([r["final_mse"] for r in g])
        trap = d["meta"]["trap"]
        for p, h in [("greedy", 0)] + [(spec(h), h) for h in horizons]:
            if p not in planners:
                continue
            rs = planners[p]
            avg = np.array([r["avg_mse"] for r in rs])
            fin = np.array([r["final_mse"] for r in rs])
            d_avg = 100 * (avg - g_avg) / g_avg.mean()  # % of greedy's mean, paired per run
            d_fin = 100 * (fin - g_fin) / g_fin.mean()
            ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            rows.append({"condition": name, "planner": p, "horizon": h, "runs": len(rs),
                         "avg_mse": avg.mean(), "d_avg_pct": d_avg.mean(), "d_avg_ci": ci(d_avg),
                         "final_mse": fin.mean(), "d_final_pct": d_fin.mean(), "d_final_ci": ci(d_fin),
                         "worst_avg_mse": avg.max(), "p90_avg_mse": np.percentile(avg, 90),
                         "stalled": sum(r["stall"] >= trap for r in rs) / len(rs),
                         "cpu_per_step": np.mean([r["cpu_per_step"] for r in rs])})
    return rows


def plot(rows, conditions, horizons):
    import matplotlib.pyplot as plt

    names = [c for c in conditions if any(r["condition"] == c for r in rows)]
    if not names:
        return
    metrics = [("d_avg_pct", "d_avg_ci", "avg MSE vs greedy (%)"),
               ("d_final_pct", "d_final_ci", "final MSE vs greedy (%)"),
               ("stalled", None, "stalled runs (share)"),
               ("cpu_per_step", None, "CPU s per step (team)")]
    fig, axs = plt.subplots(len(metrics), len(names), figsize=(3.4 * len(names), 9.5), squeeze=False,
                            sharex=True)
    for c, name in enumerate(names):
        rs = [r for r in rows if r["condition"] == name]
        g = next(r for r in rs if r["horizon"] == 0)
        mc = sorted((r for r in rs if r["horizon"] > 0), key=lambda r: r["horizon"])
        x = [r["horizon"] for r in mc]
        for m, (key, ci, label) in enumerate(metrics):
            ax = axs[m, c]
            y = [r[key] for r in mc]
            if ci:
                ax.errorbar(x, y, yerr=[r[ci] for r in mc], fmt="o-", color="C0", capsize=3, label="Dec-MCTS")
                ax.axhline(0, color="k", ls="--", lw=0.8, label="greedy")
            else:
                ax.plot(x, y, "o-", color="C0", label="Dec-MCTS")
                ax.axhline(g[key], color="k", ls="--", lw=0.8, label="greedy")
            ax.grid(alpha=0.4)
            if c == 0:
                ax.set_ylabel(label)
        axs[0, c].set_title((conditions[name][4]), fontsize=10)
        axs[-1, c].set(xscale="log", xlabel="Dec-MCTS horizon (steps)")
        axs[-1, c].set_xticks(horizons, [str(h) for h in horizons])
        axs[-1, c].set_xticks([], minor=True)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("No single horizon is best (negative = better than greedy; error bars = 95 % CI, paired runs)",
                 fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(FIGS / f"horizon_sweep.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", help=f"subset of {list(CONDITIONS) + list(EXTRA)}")
    ap.add_argument("--extra", action="store_true", help="also the full-sharing condition")
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    ap.add_argument("--runs", type=int, help="default: the configs' 20")
    ap.add_argument("--steps", type=int, help="override every condition's steps (smoke tests)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun conditions that already have results")
    args = ap.parse_args()
    every = {**CONDITIONS, **EXTRA}
    conds = {k: every[k] for k in args.conditions} if args.conditions else \
        {**CONDITIONS, **(EXTRA if args.extra else {})}
    if not args.plot_only:
        run(conds, args.horizons, args.jobs, args.runs, args.steps, args.force)
    rows = summarise(conds, args.horizons)
    if rows:
        FIGS.mkdir(parents=True, exist_ok=True)
        with open(FIGS / "horizon_sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        plot(rows, conds, args.horizons)
        print(f"-> {FIGS}/horizon_sweep.pdf, .png, .csv")


if __name__ == "__main__":
    main()
