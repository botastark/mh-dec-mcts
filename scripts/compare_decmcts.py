"""Paired comparison: Dec-MCTS vs greedy IG / IGd on the SAME fields and starts.

Run-to-run spread (different fields) is much larger than planner differences, so compare per run
(planner - ig) and report mean +- standard error of that difference.

    python scripts/compare_decmcts.py --n 1 --planners ig decmcts --runs 20 --steps 40
    python scripts/compare_decmcts.py --n 4 --range 16 --planners ig igd decmcts --runs 10 --steps 40
    python scripts/compare_decmcts.py --n 1 --planners ig decmcts --set horizon=3 --set iterations=60

Saves to results/compare/<config>_N<n>_R<range>[_<sets>]/:
    runs.npz       per planner and run: entropy, MSE, altitude curves  (e.g. H_decmcts: runs x steps)
    curves.png     mean entropy / MSE / altitude, and the paired difference to the first planner
                   with a +-2 s.e. band (outside the band = a real difference)
    run0_<p>.png   legacy viewer of run 0: field, final belief (agent 0), paths
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from mh_dec_mcts.planning.dec_mcts import make_dec_mcts
from mh_dec_mcts.planning.planners import make_planner
from mh_dec_mcts.scenario import load_config, make_scenario, make_streams, mapping_weights
from mh_dec_mcts.sim.multi import run_multi
from mh_dec_mcts.sim.single import run_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decmcts.yaml")
    ap.add_argument("--planners", nargs="+", default=["ig", "decmcts"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--range", type=int, default=16)
    ap.add_argument("--weights", help="override run.weights")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--set", action="append", default=[], help="mcts override, e.g. horizon=3")
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args()

    over = {"agents": {"n": args.n}, "comms": {"range": args.range, "sharing": "none"}}
    if args.weights:
        over["run"] = {"weights": args.weights}
    cfg = load_config(args.config, **over)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg["mcts"][k] = yaml.safe_load(v)
    print("mcts:", cfg["mcts"])

    res, first = {}, {}
    for name in args.planners:
        streams = make_streams(cfg)  # same fields and starts for every planner
        h, m, alt, t0 = [], [], [], time.time()
        for i in range(args.runs):
            sc = make_scenario(cfg, i, streams)
            if args.n == 1:
                pl = make_dec_mcts(cfg, sc) if name == "decmcts" else make_planner(name, sc.lattice, sc.rates, cfg["compat"])
                r = run_single(sc, pl, args.steps, mapping_weights(cfg))
                h.append(r.entropy), m.append(r.mse), alt.append(r.altitude)
                if i == 0:
                    first[name] = (sc, r.belief, [r.poses])
            else:
                dec = make_dec_mcts(cfg, sc) if name == "decmcts" else None
                tie = 0.0 if cfg["compat"] == "pierdicca" else 1e-4
                r = run_multi(sc, name, args.steps, mapping_weights(cfg), "none", args.range, tie, dec)
                h.append(np.mean(r.entropy, 1)), m.append(np.mean(r.mse, 1)), alt.append(np.mean(r.altitude, 1))
                if i == 0:
                    first[name] = (sc, r.beliefs[0], r.poses)
            print(f"  {name} run {i + 1}/{args.runs} ({time.time() - t0:.0f} s)", flush=True)
        res[name] = (np.array(h), np.array(m), np.array(alt))

    ref = args.planners[0]
    print(f"\nN={args.n} R={args.range} weights={mapping_weights(cfg)} {args.runs} runs x {args.steps} steps;"
          f" differences vs {ref} (negative = better), mean +- s.e.")
    for name, (h, m, _) in res.items():
        dh, dm = h - res[ref][0], m - res[ref][1]
        se = lambda x: x.std() / np.sqrt(len(x))
        print(f"{name:8s} final H {h[:, -1].mean():8.0f}  dH final {dh[:, -1].mean():+7.0f} +- {se(dh[:, -1]):5.0f}"
              f"  dH avg over steps {dh.mean(1).mean():+7.0f} +- {se(dh.mean(1)):5.0f}"
              f"  final MSE {m[:, -1].mean():.4f}  dMSE {dm[:, -1].mean():+.4f} +- {se(dm[:, -1]):.4f}")

    tag = f"{cfg['name']}_N{args.n}_R{args.range}" + "".join(f"_{kv}" for kv in args.set)
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "runs.npz", **{f"{k}_{name}": a for name, arrs in res.items() for k, a in zip("HMA", arrs)})
    plot(res, ref, out / "curves.png", f"{tag}, weights {mapping_weights(cfg)}, {args.runs} runs")
    from mh_dec_mcts.viewer import plot_terrain

    for name, (sc, belief, paths) in first.items():
        plot_terrain(out / f"run0_{name}.png", sc.lattice, sc.truth, belief, paths, title=f"{name} | {tag} | run 0")
    print(f"-> {out}")


def plot(res, ref, path, title):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, (k, label) in enumerate([(0, "Entropy (bits)"), (1, "MSE"), (2, "Altitude (m)")]):
        for name, arrs in res.items():
            a = arrs[k]
            mu, se = a.mean(0), a.std(0) / np.sqrt(len(a))
            t = np.arange(a.shape[1])
            l, = axs[0, col].plot(t, mu, label=name)
            axs[0, col].fill_between(t, mu - se, mu + se, color=l.get_color(), alpha=0.2, lw=0)
            if name != ref:
                d = a - res[ref][k]
                mu, se = d.mean(0), d.std(0) / np.sqrt(len(d))
                axs[1, col].plot(t, mu, color=l.get_color(), label=f"{name} - {ref}")
                axs[1, col].fill_between(t, mu - 2 * se, mu + 2 * se, color=l.get_color(), alpha=0.2, lw=0)
        axs[0, col].set(ylabel=label, xlabel="Step")
        axs[1, col].axhline(0, color="k", lw=0.8)
        axs[1, col].set(ylabel=f"paired difference ({label})", xlabel="Step")
        for ax in axs[:, col]:
            ax.grid(alpha=0.4)
            ax.legend(fontsize=8)
    fig.suptitle(title + " (top: mean +- s.e.; bottom: below 0 = better than " + ref + ", band +-2 s.e.)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
