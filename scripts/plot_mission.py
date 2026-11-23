"""Planner-comparison figures from a run.py result.

    python scripts/plot_mission.py results/myopia/swarm_rect_IGd_BM_R5

One set of figures per team setting (n, sharing, range) of the result:
    mission.png    row 1, the mission: MSE, entropy (mean of the UAVs' own maps), altitude,
                   coverage, planning time; row 2, the team: footprint overlap, connectivity,
                   misalignment, backfill. Each run averaged over the UAVs; line = mean over
                   runs, shade = +-1 std over runs (Pierdicca et al.'s convention).
    runs.png       one dot per run: avg MSE of each planner vs the first planner on the same run
                   (below the diagonal = better), final coverage, longest stall.
    firstseen.png  (needs run.py --save-poses) the run where the first planner did worst (lowest
                   final coverage, then highest MSE), every planner on that same field: step at
                   which each cell was first seen, and the paths (light = start, dark = end).
Also works on an old runs.json of a planner comparison (mission curves without std, runs.png).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mh_dec_mcts.experiment import load_arrays, load_results
from mh_dec_mcts.plotting import band
from mh_dec_mcts.scenario import make_lattice
from mh_dec_mcts.world.grid import Grid

PANELS = [("mse", "MSE"), ("entropy", "Entropy (bits)"), ("alt", "Altitude (m)"), ("coverage", "Coverage"),
          ("plan", "Planning time per step (CPU s, team)"), ("overlap", "Footprint overlap between UAVs"),
          ("connected", "UAVs with a teammate in range"), ("misalignment", "Belief misalignment"),
          ("backfill", "Backfill (fraction of field)")]
UAV_HUES = ("#e41a1c", "#ff7f00", "#f0027f", "#9e9e9e", "#a6561d", "#ffd92f")  # red, orange, magenta, grey, brown, gold


def groups(results: dict) -> dict[tuple, list[tuple[int, dict]]]:
    """(n, sharing, range, setting) -> [(condition index, condition)], in result order."""
    out: dict[tuple, list] = {}
    for k, c in enumerate(results["conditions"]):
        out.setdefault((c["n"], c["sharing"], c["range"], c["setting"]), []).append((k, c))
    return out


def arrays(conds: list[dict]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """planner -> panel key -> (mean, std) over runs: what mission.png plots."""
    out = {}
    for c in conds:
        have = c["runs"][0]["curves"] if c["runs"][0]["curves"] is not None else c["mean_curves"]
        out[c["planner"]] = {key: band(c, key) for key, _ in PANELS if key in have}
    return out


def plot_curves(conds: list[dict], path, title):
    fig, axs = plt.subplots(2, 5, figsize=(25, 8))
    axs[1, 4].axis("off")
    data = arrays(conds)
    for ax, (key, label) in zip(axs.flat, PANELS):
        for s, d in data.items():
            if key not in d:
                continue
            mu, sd = d[key]
            line, = ax.plot(mu, label=s)
            ax.fill_between(np.arange(len(mu)), mu - sd, mu + sd, color=line.get_color(), alpha=0.2, lw=0)
        ax.set(xlabel="Step", ylabel=label)
        ax.grid(alpha=0.4)
    axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(4, 4))  # x 1e4, as Pierdicca
    axs[0, 4].set_yscale("log")  # greedy takes milliseconds, long horizons seconds
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title + "   (top: mission, bottom: team; mean +- std over runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_runs(conds: list[dict], path, title):
    """Per run: avg MSE vs the first planner (paired), final coverage, longest stall."""
    names = [c["planner"] for c in conds]
    sc = [[r["scalars"] for r in c["runs"]] for c in conds]
    ref = np.array([r["avg_mse"] for r in sc[0]])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for k, (s, rs) in enumerate(zip(names, sc)):
        col = f"C{k}"
        if k:
            axs[0].scatter(ref, [r["avg_mse"] for r in rs], s=18, color=col, label=s)
        x = k + np.linspace(-0.2, 0.2, len(rs))
        axs[1].scatter(x, [r["final_cov"] for r in rs], s=14, color=col)
        axs[2].scatter(x, [r["stall"] for r in rs], s=14, color=col)
    lim = [min(ref.min(), axs[0].get_ylim()[0]), max(ref.max(), axs[0].get_ylim()[1])]
    axs[0].plot(lim, lim, "k--", lw=0.8)
    axs[0].set(xlabel=f"avg MSE, {names[0]}", ylabel="avg MSE, planner (same run)",
               title="below the diagonal = better than " + names[0])
    if len(names) > 1:
        axs[0].legend(fontsize=8)
    for ax, lab in ((axs[1], "final coverage"), (axs[2], "longest stall (steps)")):
        ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
        ax.set(ylabel=lab)
    for ax in axs:
        ax.grid(alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def worst_run(cond: dict) -> int:
    """The run with the lowest final coverage, then the highest avg MSE."""
    sc = [r["scalars"] for r in cond["runs"]]
    return min(range(len(sc)), key=lambda i: (sc[i]["final_cov"], -sc[i]["avg_mse"]))


def path_cmap(hue: str):
    """Light tint of `hue` (first steps) -> the hue -> a dark shade of it (last steps)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    c = np.array(to_rgb(hue))
    return LinearSegmentedColormap.from_list(hue, [c + 0.6 * (1 - c), c, 0.4 * c])


def plot_first_seen(conds: list[tuple[int, dict]], arr: dict, cfg: dict, run: int, path, steps: int):
    """Run `run` (the same field for every planner): step at which each cell was first seen, and
    the UAV paths, light at the first steps to dark at the last, one hue per UAV."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    grid = Grid(**cfg["grid"])
    lat = make_lattice(grid, cfg["lattice"])
    truth = arr[f"truth_r{run}"]
    norm = Normalize(0, steps)  # one step scale for every UAV and planner
    fig, axs = plt.subplots(len(conds), 1, figsize=(8, 3.8 * len(conds)), squeeze=False, layout="constrained")
    n_uav = 1
    for ax, (k, c) in zip(axs[:, 0], conds):
        first = arr[f"c{k}_r{run}_first"].astype(float)
        first[first < 0] = np.nan  # never seen: white
        im = ax.imshow(first, cmap="viridis", vmin=0, vmax=steps, origin="upper")
        ax.contour(truth, levels=[0.5], colors="k", linewidths=0.4, alpha=0.5)  # ground-truth clusters
        paths = arr[f"c{k}_r{run}_poses"]
        n_uav = len(paths)
        for a, poses in enumerate(paths):
            cmap = path_cmap(UAV_HUES[a % len(UAV_HUES)])
            ij = np.array([grid.xy_to_ij(*lat.xyz(_pose(p))[:2]) for p in poses], dtype=float)
            xy = ij[:, ::-1]  # (column, row) = (x, y) on the image
            lc = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap=cmap, norm=norm, linewidths=1.1)
            lc.set_array(np.arange(len(xy) - 1))  # segment t: step t -> t + 1
            ax.add_collection(lc)
            ax.plot(*xy[0], "o", ms=5, mfc=cmap(0.0), mec="k", mew=0.6)  # start
        s = c["runs"][run]["scalars"]
        ax.set(title=f"{c['planner']}: coverage {s['final_cov']:.2f}, avg MSE {s['avg_mse']:.4f}",
               xticks=[], yticks=[], xlim=(-0.5, first.shape[1] - 0.5), ylim=(first.shape[0] - 0.5, -0.5))
        fig.colorbar(im, ax=ax, shrink=0.9)
    if n_uav > 1:
        axs[0, 0].legend(handles=[Line2D([], [], color=UAV_HUES[a % len(UAV_HUES)], lw=2, label=f"UAV {a}")
                                  for a in range(n_uav)], loc="upper left", fontsize=7, ncol=min(n_uav, 4))
    fig.suptitle(f"run {run} (worst for the first planner): step at which each cell was first seen\n"
                 "white = never seen, black lines = ground-truth cluster edges\n"
                 "paths: light at the start (o) to dark at the end" + (", one hue per UAV" if n_uav > 1 else ""))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pose(p):
    from mh_dec_mcts.sensing.lattice import Pose

    return Pose(*map(int, p))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="a run.py output folder (results.json), or an old planner-comparison folder (runs.json)")
    ap.add_argument("--out", help="output folder (default: the result folder)")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    res, arr = load_results(folder), load_arrays(folder)
    out = Path(args.out) if args.out else folder
    out.mkdir(parents=True, exist_ok=True)
    meta, gs = res["meta"], groups(res)
    name = (meta.get("config_resolved") or {}).get("name") or Path(str(meta.get("config"))).stem
    for (n, sharing, range_, setting), conds in gs.items():
        suffix = "" if len(gs) == 1 else f"_n{n}_{sharing}_R{range_}"
        title = f"{name} {setting or f'sharing {sharing}, range {range_}'}, {n} UAV{'s' if n > 1 else ''}"
        cs = [c for _, c in conds]
        runs = len(cs[0]["runs"])
        plot_curves(cs, out / f"mission{suffix}.png", f"{title}: {runs} runs")
        plot_runs(cs, out / f"runs{suffix}.png", f"{title}: one dot per run")
        if arr:
            worst = worst_run(cs[0])
            plot_first_seen(conds, arr, meta["config_resolved"], worst, out / f"firstseen{suffix}.png",
                            cs[0]["steps"])
        else:
            print("no results.npz (run.py --save-poses): firstseen.png skipped")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
