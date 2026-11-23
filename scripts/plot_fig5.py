"""Pierdicca Fig. 5 layout from a run.py result over planners x sharing x ranges x team sizes.

    python scripts/run.py configs/pierdicca_fig5.yaml --planners ig igd --sharing none BS BM \\
        --ranges 5 16 --n 1 2 4 8
    python scripts/plot_fig5.py results/runs/pierdicca_fig5_r5_adaptive
    python scripts/plot_fig5.py RESULT --planners ig igd mhmcts:horizon=4

Columns: one per planner (default: every planner of the result). Rows: MSE for R = infinity
(range 16, solid), MSE for R = 5 (dashed), misalignment (R = 5, dashed). Colour = sharing (none
green, BS orange, BM blue), darker = more agents, red dots = N = 1. Insets: mean altitude.
Without sharing, IG agents ignore each other, so the green IG curves lie on the N = 1 dots (as
in the paper). Mean lines with shaded std over runs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mh_dec_mcts.experiment import load_results
from mh_dec_mcts.plotting import PAPER, band, label

SHARING = {"none": ("Greens", "green", ""), "BS": ("Oranges", "orange", " + BS"), "BM": ("Blues", "blue", " + BM")}
LINESTYLE = {16: "-", 5: "--"}  # paper: solid R = infinity, dashed R = 5
NS = (2, 4, 8)
ALTS = (5, 10, 16, 21, 27, 32)
METRICS = ("mse", "altitude", "misalignment")


def key(planner: str, sharing: str, range_, n: int) -> str:
    return f"{planner}_{sharing}_R{range_}_N{n}"  # the file names of the former Fig. 5 runner


def arrays(results: dict) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """<planner>_<sharing>_R<range>_N<n> -> metric -> (mean, std) over runs: what is plotted."""
    return {key(c["planner"], c["sharing"], c["range"], c["n"]): {m: band(c, PAPER[m]) for m in METRICS}
            for c in results["conditions"]}


def colour(sharing: str, n: int):
    return plt.get_cmap(SHARING[sharing][0])(0.35 + 0.65 * NS.index(n) / (len(NS) - 1))


def curve(ax, d, metric, color, ls, lw=1.2, shade=True):
    mu, sd = d[metric]
    t = np.arange(len(mu))
    ax.plot(t, mu, color=color, ls=ls, lw=lw)
    if shade:
        ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.06, lw=0)


def n1_dots(ax, data, planner, metric, every=5):
    d = data.get(key(planner, "none", 5, 1)) or data.get(key(planner, "none", 16, 1))
    if d:
        mu = d[metric][0]
        t = np.arange(0, len(mu), every)
        ax.plot(t, mu[t], "o", color="red", ms=2.5, zorder=5)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="default: <folder>/paper_fig5.png")
    ap.add_argument("--planners", nargs="+", help="one column each (default: every planner of the result)")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    res = load_results(folder)
    data = arrays(res)
    planners = args.planners or list(dict.fromkeys(c["planner"] for c in res["conditions"]))
    steps = res["meta"]["steps"]

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
                    d = data.get(key(planner, sharing, range_, n))
                    if d:
                        curve(ax, d, "mse", colour(sharing, n), ls)
                        curve(inset, d, "altitude", colour(sharing, n), ls, lw=1.0, shade=False)
            n1_dots(ax, data, planner, "mse")
            n1_dots(inset, data, planner, "altitude", every=10)
            ax.set(ylabel="MSE", xlabel="Step", ylim=(-0.005, 0.26), xticks=range(0, steps + 1, 10))
            inset.set(yticks=ALTS, ylim=(3, 34), xticks=(0, steps // 2, steps))
            inset.set_ylabel("h [m]", fontsize=7)
            inset.tick_params(labelsize=6)
            for a in (ax, inset):
                a.grid(alpha=0.4)
        ax = axs[2, c]
        for sharing in SHARING:
            for n in NS:
                d = data.get(key(planner, sharing, 5, n))
                if d:
                    curve(ax, d, "misalignment", colour(sharing, n), LINESTYLE[5])
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
    out = Path(args.out) if args.out else folder / "paper_fig5.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
