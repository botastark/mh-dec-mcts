"""Paper Fig. 3 layout from a run_single.py output folder.

    python scripts/run_single.py configs/paper.yaml          # 20 runs x 300 steps, ig/random/sweep
    python scripts/plot_paper_fig3.py results/single/paper_r5_equal

Top: entropy and MSE. Center: altitude and coverage (mean, shaded std over runs).
Bottom: run-0 trajectories of IG (left) and Sweep (right), first 100 steps, coloured by step.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from mh_dec_mcts.scenario import make_lattice, with_defaults
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.world.grid import Grid

COLORS = {"ig": "green", "random": "blue", "sweep": "orange"}
NAMES = {"ig": "IG", "random": "Random", "sweep": "Sweep"}
TRAJ_STEPS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="default: <folder>/paper_fig3.png")
    args = ap.parse_args()
    folder = Path(args.folder)
    data = json.loads((folder / "curves.json").read_text())
    curves, cfg = data["curves"], with_defaults(data["config"])
    run0 = np.load(folder / "run0.npz")
    lat = make_lattice(Grid(**cfg["grid"]), cfg["lattice"])

    fig = plt.figure(figsize=(8, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.3)

    panels = [("entropy", "Entropy", 0, 0), ("mse", "MSE", 0, 1),
              ("altitude", "Altitude h [m]", 1, 0), ("coverage", "Coverage", 1, 1)]
    for key, label, r, c in panels:
        ax = fig.add_subplot(gs[r, c])
        for p in ("ig", "random", "sweep"):
            if p not in curves:
                continue
            mu, sd = np.array(curves[p][key]["mean"]), np.array(curves[p][key]["std"])
            t = np.arange(len(mu))
            ax.plot(t, mu, color=COLORS[p], lw=1.2, label=NAMES[p])
            ax.fill_between(t, mu - sd, mu + sd, color="lavender" if p == "random" else COLORS[p],
                            alpha=0.6 if p == "random" else 0.15, lw=0)
        ax.set(xlabel="Step", ylabel=label, xlim=(-10, len(mu) + 5))
        ax.grid(alpha=0.4)
        if key == "entropy":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
            ax.legend(fontsize=8, loc="upper right")
        if key == "altitude":
            ax.set_yticks(np.round(lat.altitudes, 2))
            ax.set_ylim(lat.altitudes[0] - 2, lat.altitudes[-1] + 2)

    for c, p in enumerate(("ig", "sweep")):
        if f"poses_{p}" in run0:
            ax = fig.add_subplot(gs[2, c], projection="3d")
            plot_trajectory(ax, lat, run0["truth"], run0[f"poses_{p}"][: TRAJ_STEPS + 1])
            ax.set_title(NAMES[p], fontsize=9)

    sm = cm.ScalarMappable(cmap="jet", norm=Normalize(0, TRAJ_STEPS))
    cax = fig.add_axes((0.13, 0.04, 0.3, 0.008))
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("Step")

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
