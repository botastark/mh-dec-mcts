"""Plot run_swarm.py output: one column per setting, rows MSE / entropy / altitude / misalignment,
one line per planner (mean, shade = +-1 s.e. over runs). A last row shows each planner minus
greedy, paired per run, with a +-2 s.e. band (below 0 = better than greedy).

    python scripts/plot_swarm.py results/swarm/swarm
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOURS = {"greedy": "tab:green", "decmcts": "tab:blue", "mh": "tab:red"}
LABELS = {"entropy": "Entropy (bits)", "mse": "MSE", "altitude": "Altitude (m)", "misalignment": "Misalignment"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out")
    args = ap.parse_args()
    folder = Path(args.folder)
    data = {}
    for f in sorted(folder.glob("*.npz")):
        setting, planner = f.stem.split("__")
        data.setdefault(setting, {})[planner] = dict(np.load(f))
    if not data:
        raise SystemExit(f"no .npz files in {folder}")

    settings = sorted(data, reverse=True)  # IG_BS_Rinf, IGd_BM_R5
    rows = list(LABELS) + ["diff"]
    fig, axs = plt.subplots(len(rows), len(settings), figsize=(6 * len(settings), 15), squeeze=False)
    for c, setting in enumerate(settings):
        axs[0, c].set_title(setting.replace("_", " "))
        for planner, d in data[setting].items():
            col = COLOURS.get(planner)
            for r, m in enumerate(LABELS):
                a = d[m]
                mu, se = a.mean(0), a.std(0) / np.sqrt(len(a))
                t = np.arange(len(mu))
                axs[r, c].plot(t, mu, color=col, label=planner)
                axs[r, c].fill_between(t, mu - se, mu + se, color=col, alpha=0.2, lw=0)
                axs[r, c].set(ylabel=LABELS[m])
            ref = data[setting].get("greedy")
            if ref is not None and planner != "greedy":
                n = min(len(d["mse"]), len(ref["mse"]))
                diff = d["mse"][:n] - ref["mse"][:n]
                mu, se = diff.mean(0), diff.std(0) / np.sqrt(n)
                t = np.arange(len(mu))
                axs[-1, c].plot(t, mu, color=col, label=f"{planner} - greedy")
                axs[-1, c].fill_between(t, mu - 2 * se, mu + 2 * se, color=col, alpha=0.2, lw=0)
        axs[-1, c].axhline(0, color="k", lw=0.8)
        axs[-1, c].set(ylabel="MSE difference (paired)", xlabel="Step")
        for ax in axs[:, c]:
            ax.grid(alpha=0.4)
            ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out) if args.out else folder / "swarm.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
