"""Paired views: every planner minus the reference (the first planner of its team setting) on the
same runs (same field, starts and sensor noise). Differences per run, then mean and 95 % CI
(Student t) over runs; p from the Wilcoxon signed-rank test. This shows effects of a few percent
that the mean +- std bands hide (the spread between fields is much larger than between planners).

The mission is split at `split`: by default the reference's mean step to 95 % coverage, i.e.
before (exploration) and after (refinement) the field has been seen.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mh_dec_mcts.plots.common import label
from mh_dec_mcts.plots.mission import groups, title

CURVE_KEYS = ("mse", "alt", "overlap")
NAMES = {"mse": "MSE", "alt": "Altitude (m)", "overlap": "Footprint overlap", "entropy": "Entropy (bits)",
         "coverage_paper": "Coverage", "backfill": "Backfill"}
# (name, per-run value from a condition and the split step); in % of the reference's mean
METRICS = (
    ("avg MSE", lambda c, s: _scalar(c, "avg_mse")),
    ("MSE before split", lambda c, s: _window(c, 0, s)),
    ("MSE after split", lambda c, s: _window(c, s, None)),
    ("final MSE", lambda c, s: _scalar(c, "final_mse")),
    ("steps to 95 % cov.", lambda c, s: _scalar(c, "t95")),
    ("mean altitude", lambda c, s: _scalar(c, "altitude")),
    ("CPU s / step", lambda c, s: _scalar(c, "cpu_per_step")),
)


def _scalar(c: dict, key: str) -> np.ndarray:
    return np.array([np.nan if r["scalars"][key] is None else r["scalars"][key] for r in c["runs"]], float)


def _window(c: dict, lo: int, hi: int | None) -> np.ndarray:
    return np.array([np.mean(r["curves"]["mse"][lo:hi]) for r in c["runs"]])


def ci95(d: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half-width of the 95 % CI of the mean over runs (NaN-aware: runs that never reach a level)."""
    n = np.sum(~np.isnan(d), axis=axis)
    return stats.t.ppf(0.975, np.maximum(n - 1, 1)) * np.nanstd(d, axis=axis, ddof=1) / np.sqrt(n)


def split_step(ref: dict) -> int:
    t95 = _scalar(ref, "t95")
    return round(float(np.nanmean(t95))) if np.any(~np.isnan(t95)) else ref["steps"] // 2


def compare(res: dict, split: int | None = None) -> list[dict]:
    """Per team setting: (setting title, reference, [(other condition)], split step)."""
    out = []
    for g, kc in groups(res).items():
        cs = [c for _, c in kc]
        out.append({"title": title(res, g), "ref": cs[0], "others": cs[1:],
                    "split": split if split is not None else split_step(cs[0])})
    return out


def rows(blocks: list[dict]) -> list[dict]:
    """One row per setting x planner x metric: mean difference (and in % of the reference), 95 % CI,
    runs where the planner is lower, Wilcoxon p."""
    out = []
    for b in blocks:
        for c in b["others"]:
            for name, f in METRICS:
                x, y = f(c, b["split"]), f(b["ref"], b["split"])
                d = x - y
                ok = ~np.isnan(d)
                ref_mean = np.nanmean(y)
                p = stats.wilcoxon(d[ok]).pvalue if np.any(d[ok] != 0) else 1.0
                out.append({"setting": b["title"], "planner": c["planner"], "ref": b["ref"]["planner"],
                            "metric": name, "split": b["split"], "d": np.nanmean(d), "ci": ci95(d),
                            "pct": 100 * np.nanmean(d) / ref_mean, "pct_ci": 100 * ci95(d) / abs(ref_mean),
                            "lower": int(np.sum(d[ok] < 0)), "ties": int(np.sum(d[ok] == 0)),
                            "runs": int(ok.sum()), "p": p})
    return out


def table(rs: list[dict]) -> str:
    head = (f"{'setting':34s} {'planner':34s} {'metric':20s} {'diff':>10s} {'% of ref':>18s}"
            f" {'lower/ties/runs':>16s} {'p (Wilcoxon)':>12s}")
    lines = [head]
    for r in rs:
        lines.append(f"{r['setting'][:34]:34s} {r['planner'][:34]:34s} {r['metric']:20s} {r['d']:+10.5f}"
                     f" {r['pct']:+8.1f} +- {r['pct_ci']:5.1f}% {r['lower']:5d}/{r['ties']:2d}/{r['runs']:2d}"
                     f"       {r['p']:8.3f}{' *' if r['p'] < 0.05 else ''}")
    return "\n".join(lines)


def curves(blocks: list[dict], path: Path, keys=CURVE_KEYS) -> None:
    """Rows = curve keys, columns = team settings: planner - reference per step, mean +- 95 % CI."""
    fig, axs = plt.subplots(len(keys), len(blocks), figsize=(5.2 * len(blocks), 3.3 * len(keys)),
                            squeeze=False, sharex="col")
    for j, b in enumerate(blocks):
        for i, key in enumerate(keys):
            ax = axs[i, j]
            y = np.array([r["curves"][key] for r in b["ref"]["runs"]])
            for n, c in enumerate(b["others"]):
                d = np.array([r["curves"][key] for r in c["runs"]]) - y
                mu, h = d.mean(0), ci95(d)
                line, = ax.plot(mu, color=f"C{n + 1}", label=label(c["planner"]))
                ax.fill_between(np.arange(len(mu)), mu - h, mu + h, color=line.get_color(), alpha=0.2, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.axvline(b["split"], color="grey", ls="--", lw=0.8)
            ax.grid(alpha=0.4)
            if j == 0:
                ax.set_ylabel(f"{NAMES.get(key, key)}\nminus {label(b['ref']['planner'])}")
            if i == 0:
                ax.set_title(b["title"], fontsize=10)
            if i == len(keys) - 1:
                ax.set_xlabel("Step")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Paired difference to the reference per step (same runs): mean +- 95 % CI over runs;"
                 " below 0 = lower than the reference; dashed = split (reference's mean step to 95 % coverage)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def forest(rs: list[dict], path: Path) -> None:
    """One panel per metric: % difference to the reference with 95 % CI, one line per setting x
    planner; filled = Wilcoxon p < 0.05."""
    metrics = list(dict.fromkeys(r["metric"] for r in rs))
    labels = list(dict.fromkeys((r["setting"], r["planner"]) for r in rs))
    planners = list(dict.fromkeys(r["planner"] for r in rs))
    fig, axs = plt.subplots(1, len(metrics), figsize=(2.9 * len(metrics) + 3, 0.42 * len(labels) + 1.6),
                            sharey=True, squeeze=False)
    for ax, m in zip(axs[0], metrics):
        for y, (s, p) in enumerate(labels):
            r = next(r for r in rs if r["metric"] == m and r["setting"] == s and r["planner"] == p)
            col = f"C{planners.index(p) + 1}"
            ax.errorbar(r["pct"], y, xerr=r["pct_ci"], fmt="o", color=col, capsize=3,
                        mfc=col if r["p"] < 0.05 else "white")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("% vs reference")
        ax.grid(alpha=0.4, axis="x")
    axs[0, 0].set_yticks(range(len(labels)), [f"{label(p)}  |  {s}" for s, p in labels], fontsize=8)
    axs[0, 0].invert_yaxis()
    fig.suptitle("Planner vs reference on the same runs: mean +- 95 % CI over runs (filled: Wilcoxon p < 0.05;"
                 " < 0 = lower: better for MSE, steps and CPU)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
