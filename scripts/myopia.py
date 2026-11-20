"""Does greedy miss ground and come back for it? Greedy vs Dec-MCTS horizons, one swarm setting.

    python scripts/myopia.py configs/swarm_rect.yaml --setting IGd_BM_R5 \\
        --planners greedy decmcts:horizon=2 decmcts:horizon=8,rollout=greedy
    python scripts/myopia.py configs/swarm.yaml --planners greedy decmcts      # same on the square

Planner specs: greedy (the setting's IG / IGd) or decmcts[:key=value,...] (overrides of `mcts`).
Same fields, starts and sensor noise for every planner (paired). Per run, team-wide, from the
flown poses (starts excluded, they are not observed):
    coverage(t)   fraction of the field inside at least one footprint of any UAV
    backfill(t)   cumulative fraction of the field first seen more than `late` steps (default 10)
                  after any UAV first saw a cell of the same x-column: ground flown past without
                  being seen and come back for. Small for clean sweeps, grows with short-sightedness.
Output in results/myopia/<name>_<setting>/:
    myopia.png     coverage, backfill, MSE (mean +- s.e.) and the UAVs' x(t) in run 0
    firstseen.png  run 0: step at which each cell was first seen, paths on top
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from mh_dec_mcts.planning.dec_mcts import make_dec_mcts
from mh_dec_mcts.scenario import load_config, make_scenario, mapping_weights
from mh_dec_mcts.sim.multi import run_multi


def parse(spec: str) -> tuple[str, dict]:
    """'decmcts:horizon=8,rollout=greedy' -> ('decmcts', {'horizon': 8, 'rollout': 'greedy'})"""
    name, _, rest = spec.partition(":")
    return name, {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in rest.split(",") if kv)}


def one_run(cfg: dict, setting: dict, spec: str, run: int, steps: int, late: int) -> dict:
    name, over = parse(spec)
    cfg = {**cfg, "mcts": {**cfg["mcts"], **over}}
    sc = make_scenario(cfg, run)
    planner = setting["greedy"] if name == "greedy" else name
    dec = make_dec_mcts(cfg, sc) if planner == "decmcts" else None
    res = run_multi(sc, planner, steps, mapping_weights(cfg), setting["sharing"], setting["range"], 1e-4, dec)
    lat = sc.lattice
    first = np.full(sc.grid.shape, -1, dtype=np.int32)
    col_first = np.full(sc.grid.shape[1], -1)  # step at which any cell of each column was first seen
    cov, back, n_back = [], [], 0
    for t in range(steps):
        for path in res.poses:
            i0, i1, j0, j1 = lat.footprint(path[t + 1])
            win, cf = first[i0:i1, j0:j1], col_first[j0:j1]
            new = win < 0
            n_back += int(new[:, (cf >= 0) & (t - cf > late)].sum())  # columns flown over long ago
            win[new] = t
            cf[(cf < 0) & new.any(axis=0)] = t
        cov.append(float((first >= 0).mean()))
        back.append(n_back / first.size)
    keep = run == 0
    return {"coverage": cov, "backfill": back, "mse": np.mean(res.mse, axis=1).tolist(),
            "entropy": np.mean(res.entropy, axis=1).tolist(), "altitude": float(np.mean(res.altitude)),
            "x": [[lat.xyz(p)[0] for p in path[1:]] for path in res.poses] if keep else None,
            "first": first if keep else None, "poses": res.poses if keep else None, "sc": sc if keep else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--setting", default="IGd_BM_R5")
    ap.add_argument("--planners", nargs="+", default=["greedy", "decmcts:horizon=2", "decmcts:horizon=8"])
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--late", type=int, default=10, help="steps after which a first sighting counts as backfill")
    ap.add_argument("--out", default="results/myopia")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setting = cfg["settings"][args.setting]
    cfg["comms"] = {"range": setting["range"], "sharing": setting["sharing"]}
    runs, steps = args.runs, args.steps or cfg["run"]["steps"]

    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {(s, i): ex.submit(one_run, cfg, setting, s, i, steps, args.late)
                for s in args.planners for i in range(runs)}
        res = {s: [futs[(s, i)].result() for i in range(runs)] for s in args.planners}

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan when a run never reaches a level
    print(f"{cfg['name']} {args.setting}: {runs} runs x {steps} steps (mean +- s.e.; nan = never reached)")
    ref = np.array([np.mean(r["mse"]) for r in res[args.planners[0]]])
    print(f"{'planner':>36s} {'steps to 80% cov':>17s} {'to 95%':>8s} {'backfill %':>13s} {'avg MSE':>9s}"
          f" {'vs ' + args.planners[0] + ' (paired)':>26s} {'final MSE':>10s} {'mean alt':>8s}")
    for s, rs in res.items():
        cov = np.array([r["coverage"] for r in rs])
        t80 = np.array([np.argmax(c >= 0.8) if c[-1] >= 0.8 else np.nan for c in cov])
        t95 = np.array([np.argmax(c >= 0.95) if c[-1] >= 0.95 else np.nan for c in cov])
        bf = np.array([r["backfill"][-1] for r in rs]) * 100
        se = lambda a: np.nanstd(a) / np.sqrt(len(a))
        mse = np.array([r["mse"] for r in rs])
        d = mse.mean(1) - ref
        print(f"{s:>36s} {np.nanmean(t80):9.0f} +- {se(t80):3.0f} {np.nanmean(t95):6.0f}"
              f" {bf.mean():7.1f} +- {se(bf):3.1f} {mse.mean():9.4f}"
              f" {d.mean():+.4f} +- {se(d):.4f} ({100 * d.mean() / ref.mean():+5.1f}%)"
              f" {mse[:, -1].mean():10.5f} {np.mean([r['altitude'] for r in rs]):7.1f}m")

    out = Path(args.out) / f"{cfg['name']}_{args.setting}"
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(res, out / "myopia.png", f"{cfg['name']} {args.setting}: {runs} runs")
    plot_first_seen(res, out / "firstseen.png", steps)
    print(f"-> {out}")


def plot_curves(res, path, title):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for s, rs in res.items():
        for ax, key, label in zip(axs, ("coverage", "backfill", "mse"),
                                  ("Coverage", "Backfill (fraction of field)", "MSE")):
            a = np.array([r[key] for r in rs])
            mu, se = a.mean(0), a.std(0) / np.sqrt(len(a))
            l, = ax.plot(mu, label=s)
            ax.fill_between(np.arange(len(mu)), mu - se, mu + se, color=l.get_color(), alpha=0.2, lw=0)
            ax.set(xlabel="Step", ylabel=label)
        for k, x in enumerate(rs[0]["x"]):
            axs[3].plot(x, color=l.get_color(), lw=1, label=s if k == 0 else None)
    axs[3].set(xlabel="Step", ylabel="x of each UAV (m), run 0")
    for ax in axs:
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_first_seen(res, path, steps):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(res), 1, figsize=(8, 3.8 * len(res)), squeeze=False, layout="constrained")
    for ax, (s, rs) in zip(axs[:, 0], res.items()):
        r = rs[0]
        sc, first = r["sc"], r["first"].astype(float)
        first[first < 0] = np.nan  # never seen: white
        im = ax.imshow(first, cmap="viridis", vmin=0, vmax=steps, origin="upper")
        for poses in r["poses"]:
            ij = np.array([sc.grid.xy_to_ij(*sc.lattice.xyz(p)[:2]) for p in poses])
            ax.plot(ij[:, 1], ij[:, 0], color="red", lw=0.8)
            ax.plot(ij[0, 1], ij[0, 0], "wo", ms=5)
        ax.set(title=f"{s} (run 0)", xticks=[], yticks=[],
               xlim=(-0.5, first.shape[1] - 0.5), ylim=(first.shape[0] - 0.5, -0.5))  # no autoscale margins
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.suptitle("step at which each cell was first seen (white = never)\n"
                 "late colours behind the paths = missed, came back")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
