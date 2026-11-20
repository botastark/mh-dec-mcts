"""Paper Fig. 4 layout from the 18 run_single.py output folders (IG, r = 0..5 x w = equal/biased/adaptive).

    for w in equal biased adaptive; do for r in 0 1 2 3 4 5; do
      python scripts/run_single.py configs/pierdicca_fig4.yaml --planners ig --weights $w --r $r
    done; done
    python scripts/plot_paper_fig4.py results/single

Top: entropy and MSE, all 18 curves (colour = w, darker = larger r).
Middle: altitude, one panel per w. Bottom: coverage, one panel per w.
Mean lines with shaded std over runs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

WEIGHTS = ("equal", "biased", "adaptive")
CMAPS = {"equal": "Greens", "biased": "Blues", "adaptive": "Oranges"}
LEGEND_COLORS = {"equal": "green", "biased": "blue", "adaptive": "orange"}
RS = (0, 1, 2, 3, 4, 5)
ALTITUDES = (5.41, 10.83, 16.24, 21.65, 27.06, 32.48)


def shade(w: str, r: int):
    """Lighter for small r, darker for large r (as in the paper)."""
    return plt.get_cmap(CMAPS[w])(0.3 + 0.7 * r / max(RS))


def load(root: Path, name: str):
    data = {}
    for w in WEIGHTS:
        for r in RS:
            f = root / f"{name}_r{r}_{w}" / "curves.json"
            if not f.exists():
                raise SystemExit(f"missing {f}: run run_single.py for w={w}, r={r} first")
            data[w, r] = json.loads(f.read_text())["curves"]["ig"]
    return data


def band(ax, d, key, color, lw=1.2):
    mu, sd = np.array(d[key]["mean"]), np.array(d[key]["std"])
    t = np.arange(len(mu))
    ax.plot(t, mu, color=color, lw=lw)
    ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.08, lw=0)
    return len(mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder that holds <name>_r{r}_{w}/curves.json")
    ap.add_argument("--name", default="pierdicca_fig4")
    ap.add_argument("--out", default=None, help="default: <root>/paper_fig4.png")
    args = ap.parse_args()
    root = Path(args.root)
    data = load(root, args.name)

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 0.8, 0.8], hspace=0.45, wspace=0.9)

    # top: entropy and MSE, all conditions
    for c, (key, label) in enumerate((("entropy", "Entropy"), ("mse", "MSE"))):
        ax = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
        for w in WEIGHTS:
            for r in RS:
                n = band(ax, data[w, r], key, shade(w, r))
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
                n = band(ax, data[w, r], key, shade(w, r))
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

    out = Path(args.out) if args.out else root / "paper_fig4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
