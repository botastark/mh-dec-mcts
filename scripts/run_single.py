"""Single-agent experiments from a config (paper Fig. 3 / Fig. 4) + legacy-viewer figure of run 0.

    python scripts/run_single.py configs/paper.yaml                          # Fig. 3: ig, random, sweep
    python scripts/run_single.py configs/paper.yaml --planners ig --weights adaptive --r 2
    python scripts/run_single.py configs/paper.yaml --runs 2 --steps 30      # quick
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mh_dec_mcts.planning.dec_mcts import make_dec_mcts
from mh_dec_mcts.planning.planners import make_planner
from mh_dec_mcts.scenario import load_config, make_scenario, make_streams, mapping_weights
from mh_dec_mcts.sim.single import METRICS, run_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--planners", nargs="+", default=["ig", "random", "sweep"])
    ap.add_argument("--weights", choices=["none", "equal", "biased", "adaptive"], help="override run.weights")
    ap.add_argument("--r", type=float, help="override field.r (Gaussian cluster radius)")
    ap.add_argument("--runs", type=int, help="override run.runs")
    ap.add_argument("--steps", type=int, help="override run.steps")
    ap.add_argument("--start", choices=["paper", "legacy", "random"], help="override agents.start")
    ap.add_argument("--out", default="results/single")
    args = ap.parse_args()

    over = {"run": {k: v for k, v in (("weights", args.weights), ("runs", args.runs), ("steps", args.steps)) if v}}
    if args.start:
        over["agents"] = {"start": args.start}
    if args.r is not None:
        over["field"] = {"r": args.r}
    cfg = load_config(args.config, **over)
    rc = cfg["run"]
    tag = cfg["name"] + (f"_r{cfg['field']['r']:g}" if "r" in cfg["field"] else "") + f"_{rc['weights']}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    curves = {p: {m: [] for m in METRICS} for p in args.planners}
    first = {}
    for name in args.planners:  # one experiment per planner, as in Pierdicca's code
        streams = make_streams(cfg)  # None unless compat: pierdicca
        for i in range(rc["runs"]):
            sc = make_scenario(cfg, run=i, streams=streams)
            planner = (make_dec_mcts(cfg, sc) if name == "decmcts"
                       else make_planner(name, sc.lattice, sc.rates, cfg["compat"]))
            res = run_single(sc, planner, rc["steps"], mapping_weights(cfg))
            for m in METRICS:
                curves[name][m].append(getattr(res, m))
            if i == 0:
                first[name] = (sc, res)
            print(f"{name} run {i + 1}/{rc['runs']}: MSE {res.mse[-1]:.5f}, altitude {res.altitude[-1]:.2f}")

    summary = {p: {m: {"mean": np.mean(curves[p][m], 0).tolist(), "std": np.std(curves[p][m], 0).tolist()}
                   for m in METRICS} for p in args.planners}
    (out / "curves.json").write_text(json.dumps({"config": cfg, "planners": args.planners, "curves": summary}))
    plot_curves(summary, out / "curves.png", f"{tag}, {rc['runs']} runs")
    np.savez_compressed(out / "run0.npz", truth=first[args.planners[0]][0].truth,
                        **{f"poses_{n}": np.array([(p.ix, p.iy, p.k) for p in r.poses]) for n, (_, r) in first.items()})

    from mh_dec_mcts.viewer import plot_terrain

    for name, (sc, res) in first.items():
        plot_terrain(out / f"run0_{name}.png", sc.lattice, sc.truth, res.belief, [res.poses],
                     title=f"{name} | {tag} | run 0, step {rc['steps']}")
    print(f"-> {out}")


def plot_curves(summary, path, title):
    import matplotlib.pyplot as plt

    labels = {"entropy": "Entropy (bits)", "mse": "MSE", "altitude": "Altitude (m)", "coverage": "Coverage"}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m in zip(axs.flat, METRICS):
        for p, d in summary.items():
            mu, sd = np.array(d[m]["mean"]), np.array(d[m]["std"])
            t = np.arange(len(mu))
            ax.plot(t, mu, label=p.upper() if p == "ig" else p.capitalize())
            ax.fill_between(t, mu - sd, mu + sd, alpha=0.2)
        ax.set(xlabel="step", ylabel=labels[m])
        ax.grid(alpha=0.3)
    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
