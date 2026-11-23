"""Pierdicca Fig. 4 layout from run.py results, one per (weights, r): IG, r = 0..5 x w = equal /
biased / adaptive.

    for w in equal biased adaptive; do for r in 0 1 2 3 4 5; do
      python scripts/run.py configs/pierdicca_fig4.yaml --planners ig --weights $w --r $r --out results/fig4
    done; done
    python scripts/plot_fig4.py results/fig4

Arguments: result folders, or folders that hold them (one level down). Each result's weights and r
come from its resolved config. --planner picks the planner (default ig).
Top: entropy and MSE, all curves (colour = w, darker = larger r).
Middle: altitude, one panel per w. Bottom: coverage (Pierdicca's), one panel per w.
Mean lines with shaded std over runs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mh_dec_mcts.experiment import load_results, result_file
from mh_dec_mcts.plotting import PAPER, band, find

WEIGHTS = ("equal", "biased", "adaptive")
CMAPS = {"equal": "Greens", "biased": "Blues", "adaptive": "Oranges"}
LEGEND_COLORS = {"equal": "green", "biased": "blue", "adaptive": "orange"}
RS = (0, 1, 2, 3, 4, 5)
ALTITUDES = (5.41, 10.83, 16.24, 21.65, 27.06, 32.48)
METRICS = ("entropy", "mse", "altitude", "coverage")


def shade(w: str, r: int):
    """Lighter for small r, darker for large r (as in the paper)."""
    return plt.get_cmap(CMAPS[w])(0.3 + 0.7 * r / max(RS))


def load(paths: list[Path], planner: str) -> dict[tuple[str, float], dict]:
    """(weights, r) -> the planner's condition, from every result found in `paths`."""
    folders = []
    for p in paths:
        folders += [p] if result_file(p) else sorted(d for d in p.iterdir() if d.is_dir() and result_file(d))
    data = {}
    for f in folders:
        res = load_results(f)
        cfg = res["meta"]["config_resolved"]
        hits = find(res, planner=planner)
        if cfg and hits and "r" in cfg["field"]:
            data[cfg["run"]["weights"], cfg["field"]["r"]] = hits[0][1]
    return data


def arrays(data: dict) -> dict[tuple[str, float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """(weights, r) -> Pierdicca metric -> (mean, std) over runs: what the panels plot."""
    return {k: {m: band(c, PAPER[m]) for m in METRICS} for k, c in data.items()}


def curve(ax, mu_sd, color, lw=1.2):
    mu, sd = mu_sd
    t = np.arange(len(mu))
    ax.plot(t, mu, color=color, lw=lw)
    ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.08, lw=0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="result folders, or folders holding them")
    ap.add_argument("--planner", default="ig")
    ap.add_argument("--out", default=None, help="default: <first path>/paper_fig4.png")
    args = ap.parse_args(argv)
    data = arrays(load(args.paths, args.planner))
    missing = [(w, r) for w in WEIGHTS for r in RS if (w, r) not in data]
    if missing:
        print(f"missing (weights, r), not plotted: {missing}")
    if not data:
        raise SystemExit("no results for this planner with weights and r in its config")
    n = max(len(d["mse"][0]) for d in data.values())  # steps per run: the x-axis length

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 0.8, 0.8], hspace=0.45, wspace=0.9)

    # top: entropy and MSE, all conditions
    for c, (key, label) in enumerate((("entropy", "Entropy"), ("mse", "MSE"))):
        ax = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
        for w in WEIGHTS:
            for r in RS:
                if (w, r) in data:
                    curve(ax, data[w, r][key], shade(w, r))
        ax.set(xlabel="Step", ylabel=label, xlim=(-2, n + 1), xticks=range(0, n + 1, 10))
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

    # middle: altitude per w; bottom: coverage per w
    for row, (key, label) in ((1, ("altitude", "Altitude h [m]")), (2, ("coverage", "Coverage"))):
        for c, w in enumerate(WEIGHTS):
            ax = fig.add_subplot(gs[row, 2 * c:2 * c + 2])
            for r in RS:
                if (w, r) in data:
                    curve(ax, data[w, r][key], shade(w, r))
            ax.set(xlim=(-2, n + 1), xticks=(0, n // 2, n))
            if key == "altitude":
                ax.set_yticks(ALTITUDES)
                ax.set_ylim(ALTITUDES[0] - 2, ALTITUDES[-1] + 2)
                ax.tick_params(labelsize=7)
            else:
                ax.set_ylim(-0.02, 1.05)
                ax.set_xlabel("Step")
            if c == 0:
                ax.set_ylabel(label)
            ax.grid(alpha=0.4)

    out = Path(args.out) if args.out else args.paths[0] / "paper_fig4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
