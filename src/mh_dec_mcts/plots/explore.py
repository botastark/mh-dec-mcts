"""Analysis views of saved results: any stored curve, and the UAVs one by one."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mh_dec_mcts.experiment import CURVES, curve_stats
from mh_dec_mcts.plots.common import band, label

UAV_KEYS = ("mse", "entropy", "alt")  # per-UAV curves in results.npz (c<k>_r<i>_uav_<key>)
NAMES = {"mse": "MSE", "entropy": "Entropy (bits)", "alt": "Altitude (m)", "team_entropy": "Entropy of the team map (bits)",
         "coverage": "Coverage", "coverage_paper": "Coverage (Pierdicca)", "plan": "Planning time per step (CPU s)",
         "misalignment": "Belief misalignment", "overlap": "Footprint overlap", "connected": "UAVs with a teammate in range",
         "backfill": "Backfill (fraction of field)"}


def cond_label(c: dict, others: list[dict]) -> str:
    """The planner, plus whatever else differs between the plotted conditions (n, sharing, range)."""
    parts = [label(c["planner"])]
    for f, fmt in (("n", "N={}"), ("sharing", "{}"), ("range", "R={}")):
        if len({o[f] for o in others}) > 1:
            parts.append(fmt.format(c[f]))
    return ", ".join(parts)


def curves(conds: list[dict], keys: list[str], path: Path, title: str, log: list[str] = ()) -> None:
    """One panel per curve key, one line per condition: mean +- std over runs."""
    bad = [k for k in keys if k not in CURVES]
    if bad:
        raise SystemExit(f"unknown curve(s) {bad}; stored: {', '.join(CURVES)}")
    fig, axs = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axs[0], keys):
        for c in conds:
            mu, sd = band(c, key)
            line, = ax.plot(mu, label=cond_label(c, conds))
            ax.fill_between(np.arange(len(mu)), mu - sd, mu + sd, color=line.get_color(), alpha=0.2, lw=0)
        ax.set(xlabel="Step", ylabel=NAMES.get(key, key))
        if key in log:
            ax.set_yscale("log")
        ax.grid(alpha=0.4)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title + "   (mean +- std over runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def uavs(cond: dict, k: int, arr: dict, path: Path, title: str, keys=UAV_KEYS) -> None:
    """One panel per metric, one line per UAV: its own value, mean +- std over runs."""
    runs = [r["run"] for r in cond["runs"]]
    fig, axs = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axs[0], keys):
        per_run = np.stack([arr[f"c{k}_r{i}_uav_{key}"] for i in runs])  # runs x steps x UAVs
        for a in range(per_run.shape[2]):
            mu, sd = curve_stats(per_run[:, :, a])
            line, = ax.plot(mu, label=f"UAV {a}")
            ax.fill_between(np.arange(len(mu)), mu - sd, mu + sd, color=line.get_color(), alpha=0.15, lw=0)
        ax.set(xlabel="Step", ylabel=NAMES.get(key, key))
        ax.grid(alpha=0.4)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title}: each UAV's own map (mean +- std over runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
