"""Legacy-style terrain figure (port of multi-horizon viewer.plot_terrain).

One row, four panels:
  1. 3D terrain with UAV paths (start o, current ^) and current footprints
  2. Top view of trajectories coloured by time step
  3. Belief map
  4. Ground truth in (i, j) with current footprints
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import Normalize

from mh_dec_mcts.sensing.lattice import Lattice, Pose

AGENT_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
GT_CMAP = colors.ListedColormap(["lemonchiffon", "darkgreen"])
BELIEF_CMAP = colors.LinearSegmentedColormap.from_list("GreenYellow", ["lemonchiffon", "darkgreen"])


def plot_terrain(save_path, lat: Lattice, truth: np.ndarray, belief: np.ndarray,
                 paths: list[list[Pose]], title: str = ""):
    """paths: one list of poses per agent (oldest first)."""
    g = lat.grid
    x_range = (lat.x0, lat.x0 + g.size_x)
    y_range = (lat.y0, lat.y0 + g.size_y)
    multi = len(paths) > 1
    xyz = [np.array([lat.xyz(p) for p in path]) for path in paths]

    fig = plt.figure(figsize=(20, 5.5))
    if title:
        fig.suptitle(title)

    # ---- 1: 3D terrain with UAV paths ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    s = max(1, max(g.shape) // 100)  # draw at most ~100x100 facets
    rows, cols = g.shape
    xs = lat.x0 + (np.arange(0, cols, s) + 0.5) * g.cell
    ys = y_range[1] - (np.arange(0, rows, s) + 0.5) * g.cell  # row 0 is the north edge
    X, Y = np.meshgrid(xs, ys)
    face = np.where(truth[::s, ::s] == 0, "yellow", "darkgreen")
    ax1.plot_surface(X, Y, np.zeros_like(X), facecolors=face, alpha=0.3, edgecolor="none", shade=False)
    for a, P in enumerate(xyz):
        col = AGENT_COLORS[a % len(AGENT_COLORS)] if multi else None
        if multi:
            ax1.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=2, alpha=0.9)
        else:
            cmap = plt.get_cmap("cool")
            for t in range(len(P) - 1):
                ax1.plot(P[t:t + 2, 0], P[t:t + 2, 1], P[t:t + 2, 2], color=cmap(t / max(1, len(P) - 1)), lw=2)
        ax1.scatter(*P[0], color=col or "k", s=50, marker="o", edgecolors="white", linewidths=1)
        ax1.scatter(*P[-1], color=col or "k", s=80, marker="^", edgecolors="white", linewidths=1,
                    label=f"Agent {a}" if multi else None)
        i0, i1, j0, j1 = lat.footprint(paths[a][-1])
        (fx0, fy1), (fx1, fy0) = g.ij_to_xy(i0, j0), g.ij_to_xy(i1, j1)
        ax1.plot([fx0, fx1, fx1, fx0, fx0], [fy0, fy0, fy1, fy1, fy0], [0.01] * 5, color=col or "red", lw=1)
    if multi:
        ax1.legend(loc="upper left", fontsize=8)
    ax1.set(xlim=x_range, ylim=y_range, zlim=(0, max(35, lat.altitudes[-1])),
            xlabel="X (m)", ylabel="Y (m)", zlabel="Altitude (m)", title="3D Terrain & UAV Paths")

    # ---- 2: top-view trajectories coloured by time ----
    ax2 = fig.add_subplot(1, 4, 2)
    tmap = plt.get_cmap("viridis")
    T = max(len(P) for P in xyz) - 1
    for a, P in enumerate(xyz):
        for t in range(len(P) - 1):
            ax2.plot(P[t:t + 2, 0], P[t:t + 2, 1], color=tmap(t / max(1, T)), lw=2)
        ax2.scatter(P[0, 0], P[0, 1], color=tmap(0), s=80, marker="o", edgecolors="black", zorder=9,
                    label="Start" if a == 0 else None)
        ax2.scatter(P[-1, 0], P[-1, 1], color=tmap((len(P) - 1) / max(1, T)), s=120, marker="^",
                    edgecolors="black", linewidths=1.5, zorder=10)
        if multi:
            ax2.annotate(f"A{a}", (P[-1, 0], P[-1, 1]), xytext=(5, 5), textcoords="offset points",
                         fontsize=9, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=tmap, norm=Normalize(0, T))
    fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04).set_label("Time Step", fontsize=9)
    ax2.set(xlim=x_range, ylim=y_range, xlabel="X (m)", ylabel="Y (m)", title="UAV Trajectories (Top View)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    # ---- 3: belief ----
    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(belief, cmap=BELIEF_CMAP, origin="upper", vmin=0, vmax=1)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set(xlabel="j-axis", ylabel="i-axis", title="Belief Map M" + (" (Fused)" if multi else ""))

    # ---- 4: ground truth (i, j) with current footprints ----
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(truth, cmap=GT_CMAP, vmin=0, vmax=1, origin="upper")
    for a, path in enumerate(paths):
        i0, i1, j0, j1 = lat.footprint(path[-1])
        ax4.plot([j0, j0, j1, j1, j0], [i0, i1, i1, i0, i0],
                 color=AGENT_COLORS[a % len(AGENT_COLORS)] if multi else "red", lw=0.9)
    ax4.set(xlim=(0, truth.shape[1]), ylim=(truth.shape[0], 0),
            xlabel="j-axis", ylabel="i-axis", title="Ground Truth (i,j)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
