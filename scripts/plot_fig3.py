"""Pierdicca Fig. 3 layout from a run.py result with one UAV.

    python scripts/run.py configs/pierdicca_fig3.yaml --planners ig random sweep --save-poses
    python scripts/plot_fig3.py results/runs/pierdicca_fig3_r5_none

Top: entropy and MSE. Center: altitude and coverage (Pierdicca's coverage; mean, shaded std over
runs). Bottom: run-0 trajectories of two planners (default IG and Sweep, if present; needs
--save-poses), first 100 steps, coloured by step. Any planner of the result is plotted; IG,
Random and Sweep keep Pierdicca's colours.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from mh_dec_mcts.experiment import load_arrays, load_results
from mh_dec_mcts.plotting import PAPER, band, colors, label
from mh_dec_mcts.scenario import make_lattice, with_defaults
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.world.grid import Grid

PANELS = [("entropy", "Entropy", 0, 0), ("mse", "MSE", 0, 1),
          ("altitude", "Altitude h [m]", 1, 0), ("coverage", "Coverage", 1, 1)]
TRAJ_STEPS = 100


def arrays(results: dict) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """planner -> Pierdicca metric -> (mean, std) over runs: what the four curve panels plot."""
    return {c["planner"]: {m: band(c, PAPER[m]) for m, *_ in PANELS} for c in results["conditions"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--traj", nargs=2, metavar="PLANNER", help="trajectory panels (default: ig sweep)")
    ap.add_argument("--out", default=None, help="default: <folder>/paper_fig3.png")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    res, arr = load_results(folder), load_arrays(folder)
    cfg = with_defaults(res["meta"]["config_resolved"])
    lat = make_lattice(Grid(**cfg["grid"]), cfg["lattice"])
    data = arrays(res)
    col = colors(list(data))

    fig = plt.figure(figsize=(8, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.3)
    for key, ylabel, r, c in PANELS:
        ax = fig.add_subplot(gs[r, c])
        for p, d in data.items():
            mu, sd = d[key]
            t = np.arange(len(mu))
            ax.plot(t, mu, color=col[p], lw=1.2, label=label(p))
            ax.fill_between(t, mu - sd, mu + sd, color="lavender" if p == "random" else col[p],
                            alpha=0.6 if p == "random" else 0.15, lw=0)
        ax.set(xlabel="Step", ylabel=ylabel, xlim=(-10, len(mu) + 5))
        ax.grid(alpha=0.4)
        if key == "entropy":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
            ax.legend(fontsize=8, loc="upper right")
        if key == "altitude":
            ax.set_yticks(np.round(lat.altitudes, 2))
            ax.set_ylim(lat.altitudes[0] - 2, lat.altitudes[-1] + 2)

    planners = [c["planner"] for c in res["conditions"]]
    traj = args.traj or [p for p in ("ig", "sweep") if p in planners] or planners[:2]
    if arr:
        for c, p in enumerate(traj[:2]):
            k = planners.index(p)
            ax = fig.add_subplot(gs[2, c], projection="3d")
            plot_trajectory(ax, lat, arr["truth_r0"], arr[f"c{k}_r0_poses"][0][: TRAJ_STEPS + 1])
            ax.set_title(label(p), fontsize=9)
        sm = cm.ScalarMappable(cmap="jet", norm=Normalize(0, TRAJ_STEPS))
        cax = fig.add_axes((0.13, 0.04, 0.3, 0.008))
        fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("Step")
    else:
        print("no results.npz (run.py --save-poses): trajectory panels skipped")

    out = Path(args.out) if args.out else folder / "paper_fig3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")


def plot_trajectory(ax, lat, truth, poses):
    g = lat.grid
    s = max(1, max(g.shape) // 100)
    xs = lat.x0 + (np.arange(0, g.shape[1], s) + 0.5) * g.cell
    ys = lat.y0 + g.size_y - (np.arange(0, g.shape[0], s) + 0.5) * g.cell
    X, Y = np.meshgrid(xs, ys)
    face = np.where(truth[::s, ::s] == 1, "green", "white")
    ax.plot_surface(X, Y, np.zeros_like(X), facecolors=face, shade=False, edgecolor="none")
    P = np.array([lat.xyz(Pose(*map(int, p))) for p in poses])
    cmap = plt.get_cmap("jet")
    for t in range(len(P) - 1):
        ax.plot(P[t:t + 2, 0], P[t:t + 2, 1], P[t:t + 2, 2], color=cmap(t / TRAJ_STEPS), lw=1.5)
    ax.set(xlim=(lat.x0, lat.x0 + g.size_x), ylim=(lat.y0, lat.y0 + g.size_y),
           zlim=(0, lat.altitudes[-1]), xlabel="x [m]", ylabel="y [m]", zlabel="h [m]")
    ax.set_zticks(np.round(np.concatenate([[0], lat.altitudes]), 1))
    ax.set_xticks([lat.x0, lat.x0 + g.size_x / 2, lat.x0 + g.size_x])
    ax.set_yticks([lat.y0, lat.y0 + g.size_y / 2, lat.y0 + g.size_y])
    ax.tick_params(labelsize=6)
    ax.view_init(elev=25, azim=-60)


if __name__ == "__main__":
    main()
