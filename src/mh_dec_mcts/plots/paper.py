"""Pierdicca et al. Figs. 3, 4 and 5 from saved results, in the paper's layout. Every figure draws
mean lines with +-1 std over runs; what it plots comes from its *_arrays function."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from mh_dec_mcts.experiment import load_results, result_file
from mh_dec_mcts.plots.common import PAPER, band, colors, find, label
from mh_dec_mcts.scenario import make_lattice, with_defaults
from mh_dec_mcts.sensing.lattice import Pose
from mh_dec_mcts.world.grid import Grid

Bands = dict[str, tuple[np.ndarray, np.ndarray]]  # metric -> (mean, std) over runs


# ------------------------------------------------------------------ Fig. 3
FIG3_PANELS = [("entropy", "Entropy", 0, 0), ("mse", "MSE", 0, 1),
               ("altitude", "Altitude h [m]", 1, 0), ("coverage", "Coverage", 1, 1)]
TRAJ_STEPS = 100


def fig3_arrays(results: dict) -> dict[str, Bands]:
    """planner -> Pierdicca metric -> (mean, std) over runs: what the four curve panels plot."""
    return {c["planner"]: {m: band(c, PAPER[m]) for m, *_ in FIG3_PANELS} for c in results["conditions"]}


def fig3(results: dict, arrays: dict, out: Path, traj: list[str] | None = None) -> None:
    """Top: entropy, MSE. Center: altitude, coverage. Bottom: run-0 trajectories of two planners
    (default IG and Sweep), first 100 steps, coloured by step."""
    cfg = with_defaults(results["meta"]["config_resolved"])
    lat = make_lattice(Grid(**cfg["grid"]), cfg["lattice"])
    data = fig3_arrays(results)
    col = colors(list(data))
    fig = plt.figure(figsize=(8, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.3)
    for key, ylabel, r, c in FIG3_PANELS:
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
    planners = [c["planner"] for c in results["conditions"]]
    traj = traj or [p for p in ("ig", "sweep") if p in planners] or planners[:2]
    if arrays:
        for c, p in enumerate(traj[:2]):
            k = planners.index(p)
            ax = fig.add_subplot(gs[2, c], projection="3d")
            _trajectory(ax, lat, arrays["truth_r0"], arrays[f"c{k}_r0_poses"][0][: TRAJ_STEPS + 1])
            ax.set_title(label(p), fontsize=9)
        sm = cm.ScalarMappable(cmap="jet", norm=Normalize(0, TRAJ_STEPS))
        cax = fig.add_axes((0.13, 0.04, 0.3, 0.008))
        fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("Step")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _trajectory(ax, lat, truth, poses):
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


# ------------------------------------------------------------------ Fig. 4
WEIGHTS = ("equal", "biased", "adaptive")
CMAPS = {"equal": "Greens", "biased": "Blues", "adaptive": "Oranges"}
LEGEND_COLORS = {"equal": "green", "biased": "blue", "adaptive": "orange"}
RS = (0, 1, 2, 3, 4, 5)
ALTITUDES = (5.41, 10.83, 16.24, 21.65, 27.06, 32.48)
FIG4_METRICS = ("entropy", "mse", "altitude", "coverage")


def fig4_load(paths: list[Path], planner: str = "ig") -> dict[tuple[str, float], dict]:
    """(weights, r) -> the planner's condition, from every result in `paths` (result folders or
    folders holding them); weights and r from each result's resolved config."""
    folders = []
    for p in map(Path, paths):
        folders += [p] if result_file(p) else sorted(d for d in p.iterdir() if d.is_dir() and result_file(d))
    data = {}
    for f in folders:
        res = load_results(f)
        cfg = res["meta"]["config_resolved"]
        hits = find(res, planner=planner)
        if cfg and hits and "r" in cfg["field"]:
            data[cfg["run"]["weights"], cfg["field"]["r"]] = hits[0][1]
    return data


def fig4_arrays(data: dict) -> dict[tuple[str, float], Bands]:
    """(weights, r) -> Pierdicca metric -> (mean, std) over runs: what the panels plot."""
    return {k: {m: band(c, PAPER[m]) for m in FIG4_METRICS} for k, c in data.items()}


def _shade(w: str, r: int):
    """Lighter for small r, darker for large r (as in the paper)."""
    return plt.get_cmap(CMAPS[w])(0.3 + 0.7 * r / max(RS))


def _curve4(ax, mu_sd, color, lw=1.2):
    mu, sd = mu_sd
    t = np.arange(len(mu))
    ax.plot(t, mu, color=color, lw=lw)
    ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.08, lw=0)


def fig4(data: dict, out: Path) -> list[tuple[str, int]]:
    """Top: entropy and MSE, all (w, r) (colour = w, darker = larger r). Middle: altitude, one
    panel per w. Bottom: coverage, one panel per w. Returns the (w, r) without results."""
    data = fig4_arrays(data)
    missing = [(w, r) for w in WEIGHTS for r in RS if (w, r) not in data]
    n = max(len(d["mse"][0]) for d in data.values())  # steps per run: the x-axis length
    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 0.8, 0.8], hspace=0.45, wspace=0.9)
    for c, (key, ylabel) in enumerate((("entropy", "Entropy"), ("mse", "MSE"))):
        ax = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
        for w in WEIGHTS:
            for r in RS:
                if (w, r) in data:
                    _curve4(ax, data[w, r][key], _shade(w, r))
        ax.set(xlabel="Step", ylabel=ylabel, xlim=(-2, n + 1), xticks=range(0, n + 1, 10))
        ax.grid(alpha=0.4)
        if key == "mse":  # paper: 0.00 .. 0.35 every 0.05
            ax.set_yticks(np.arange(0, 0.351, 0.05))
            ax.set_ylim(-0.01, 0.36)
        if key == "entropy":  # paper: 0 .. 17.5e4 every 2.5e4
            ax.set_yticks(np.arange(0, 17.51e4, 2.5e4))
            ax.set_ylim(-0.8e4, 17.5e4)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
            h_w = [Line2D([], [], color=LEGEND_COLORS[w], label=w) for w in WEIGHTS]
            h_r = [Line2D([], [], color=plt.get_cmap("Greys")(0.3 + 0.7 * r / max(RS)), label=str(r)) for r in RS]
            leg = ax.legend(handles=h_w, loc="upper right", bbox_to_anchor=(0.78, 1.0), fontsize=8, frameon=False)
            ax.add_artist(leg)
            ax.legend(handles=h_r, title="r", loc="upper right", fontsize=7, title_fontsize=8, frameon=False)
    for row, (key, ylabel) in ((1, ("altitude", "Altitude h [m]")), (2, ("coverage", "Coverage"))):
        for c, w in enumerate(WEIGHTS):
            ax = fig.add_subplot(gs[row, 2 * c:2 * c + 2])
            for r in RS:
                if (w, r) in data:
                    _curve4(ax, data[w, r][key], _shade(w, r))
            ax.set(xlim=(-2, n + 1), xticks=(0, n // 2, n))
            if key == "altitude":
                ax.set_yticks(ALTITUDES)
                ax.set_ylim(ALTITUDES[0] - 2, ALTITUDES[-1] + 2)
                ax.tick_params(labelsize=7)
            else:
                ax.set_ylim(-0.02, 1.05)
                ax.set_xlabel("Step")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.4)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return missing


# ------------------------------------------------------------------ Fig. 5
SHARING = {"none": ("Greens", "green", ""), "BS": ("Oranges", "orange", " + BS"), "BM": ("Blues", "blue", " + BM")}
LINESTYLE = {16: "-", 5: "--"}  # paper: solid R = infinity, dashed R = 5
NS = (2, 4, 8)
ALTS = (5, 10, 16, 21, 27, 32)
FIG5_METRICS = ("mse", "altitude", "misalignment")


def fig5_key(planner: str, sharing: str, range_, n: int) -> str:
    return f"{planner}_{sharing}_R{range_}_N{n}"  # the file names of the former Fig. 5 runner


def fig5_arrays(results: dict) -> dict[str, Bands]:
    """<planner>_<sharing>_R<range>_N<n> -> metric -> (mean, std) over runs: what is plotted."""
    return {fig5_key(c["planner"], c["sharing"], c["range"], c["n"]): {m: band(c, PAPER[m]) for m in FIG5_METRICS}
            for c in results["conditions"]}


def _colour5(sharing: str, n: int):
    return plt.get_cmap(SHARING[sharing][0])(0.35 + 0.65 * NS.index(n) / (len(NS) - 1))


def _curve5(ax, d, metric, color, ls, lw=1.2, shade=True):
    mu, sd = d[metric]
    t = np.arange(len(mu))
    ax.plot(t, mu, color=color, ls=ls, lw=lw)
    if shade:
        ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.06, lw=0)


def _n1_dots(ax, data, planner, metric, every=5):
    d = data.get(fig5_key(planner, "none", 5, 1)) or data.get(fig5_key(planner, "none", 16, 1))
    if d:
        mu = d[metric][0]
        t = np.arange(0, len(mu), every)
        ax.plot(t, mu[t], "o", color="red", ms=2.5, zorder=5)


def fig5(results: dict, out: Path, planners: list[str] | None = None) -> None:
    """Columns: one per planner. Rows: MSE for R = infinity (range 16, solid), MSE for R = 5
    (dashed), misalignment (R = 5). Colour = sharing, darker = more agents, red dots = N = 1.
    Insets: mean altitude."""
    data = fig5_arrays(results)
    planners = planners or list(dict.fromkeys(c["planner"] for c in results["conditions"]))
    steps = results["meta"]["steps"]
    fig, axs = plt.subplots(3, len(planners), figsize=(4.5 * len(planners), 10), height_ratios=[1.3, 1.3, 0.8],
                            squeeze=False)
    for c, planner in enumerate(planners):
        axs[0, c].set_title(label(planner))
        for r, range_ in enumerate((16, 5)):
            ls = LINESTYLE[range_]
            ax = axs[r, c]
            inset = ax.inset_axes([0.5, 0.5, 0.47, 0.45])
            for sharing in SHARING:
                for n in NS:
                    d = data.get(fig5_key(planner, sharing, range_, n))
                    if d:
                        _curve5(ax, d, "mse", _colour5(sharing, n), ls)
                        _curve5(inset, d, "altitude", _colour5(sharing, n), ls, lw=1.0, shade=False)
            _n1_dots(ax, data, planner, "mse")
            _n1_dots(inset, data, planner, "altitude", every=10)
            ax.set(ylabel="MSE", xlabel="Step", ylim=(-0.005, 0.26), xticks=range(0, steps + 1, 10))
            inset.set(yticks=ALTS, ylim=(3, 34), xticks=(0, steps // 2, steps))
            inset.set_ylabel("h [m]", fontsize=7)
            inset.tick_params(labelsize=6)
            for a in (ax, inset):
                a.grid(alpha=0.4)
        ax = axs[2, c]
        for sharing in SHARING:
            for n in NS:
                d = data.get(fig5_key(planner, sharing, 5, n))
                if d:
                    _curve5(ax, d, "misalignment", _colour5(sharing, n), LINESTYLE[5])
        ax.set(ylabel="Misalignment", xlabel="Step", xticks=range(0, steps + 1, 10))
        ax.grid(alpha=0.4)
    h_n = [Line2D([], [], ls="", marker="o", color="red", ms=3, label="1")] + \
          [Line2D([], [], color=plt.get_cmap("Greys")(0.35 + 0.65 * i / 2), label=str(n)) for i, n in enumerate(NS)]
    h_r = [Line2D([], [], color="k", ls="--", label="5"), Line2D([], [], color="k", ls="-", label="$\\infty$")]
    for c, name in enumerate(label(p) for p in planners):
        h_s = [Line2D([], [], color=SHARING[s][1], label=f"{name}{SHARING[s][2]}") for s in SHARING]
        leg = axs[0, c].legend(handles=h_s, loc="upper right", fontsize=7)
        axs[0, c].add_artist(leg)
    leg = axs[0, 0].legend(handles=h_n, title="N", loc="upper center", fontsize=7, title_fontsize=7)
    axs[0, 0].add_artist(leg)
    axs[0, 0].legend(handles=h_r, title="R", loc="center right", fontsize=7, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
