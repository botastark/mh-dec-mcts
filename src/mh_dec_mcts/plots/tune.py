"""Tuning results (scripts/tune.py): difference to greedy per parameter value, paired runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def tune(folders: list[Path], path: Path) -> None:
    """Columns = (tuning, team size, parameter); rows = avg / final MSE vs greedy, stalled, CPU."""
    cols = []
    for f in folders:
        t = json.loads((f / "tune.json").read_text())
        for n in dict.fromkeys(r["n"] for r in t["rows"]):
            rs = [r for r in t["rows"] if r["n"] == n]
            for p in dict.fromkeys(r["param"] for r in rs if r["param"] not in ("greedy", "base")):
                cols.append((f"{t['tag']}, {n} UAV{'s' if n > 1 else ''}", p, rs))
    if not cols:
        return
    metrics = [("d_avg_pct", "d_avg_ci", "avg MSE vs greedy (%)"), ("d_final_pct", "d_final_ci", "final MSE vs greedy (%)"),
               ("stalled", None, "stalled runs (share)"), ("cpu_per_step", None, "CPU s per step (team)")]
    fig, axs = plt.subplots(len(metrics), len(cols), figsize=(3.6 * len(cols), 9.5), squeeze=False, sharex="col")
    for c, (title, p, rs) in enumerate(cols):
        vals = [r for r in rs if r["param"] == p]
        greedy, base = next(r for r in rs if r["param"] == "greedy"), next(r for r in rs if r["param"] == "base")
        x = np.arange(len(vals))
        for m, (key, ci, label) in enumerate(metrics):
            ax = axs[m, c]
            y = [r[key] for r in vals]
            if ci:
                ax.errorbar(x, y, yerr=[r[ci] for r in vals], fmt="o-", color="C0", capsize=3)
                ax.axhline(0, color="k", ls="--", lw=0.8, label="greedy")
            else:
                ax.plot(x, y, "o-", color="C0")
                ax.axhline(greedy[key], color="k", ls="--", lw=0.8, label="greedy")
            ax.axhline(base[key], color="tab:gray", ls="--", lw=0.8, label="base")
            ax.grid(alpha=0.4)
            if c == 0:
                ax.set_ylabel(label)
        axs[0, c].set_title(title, fontsize=10)
        axs[-1, c].set_xticks(x, [str(r["value"]) for r in vals])
        axs[-1, c].set_xlabel(p)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Tuning: difference to greedy on paired runs (negative = better; error bars = 95 % CI)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
