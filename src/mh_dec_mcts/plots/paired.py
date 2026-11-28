"""Paired views: every planner minus the reference (by default the first planner of its team
setting; any variant with `ref`) on the same runs (same field, starts and sensor noise).
Differences per run, then mean and 95 % CI = 1.96 s.e. over runs (s.e. = std, ddof 1, / sqrt(runs));
p from the Wilcoxon signed-rank test. This shows effects of a few percent that the mean +- std
bands hide (the spread between fields is much larger than between planners).

Primary metrics (declared before the confirmation runs; `primary`): average MSE, steps to MSE 0.01
and 0.002, stall fraction, worst run, CPU per step.

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
    ("steps to MSE 0.01", lambda c, s: _t_mse(c, 0.01)),
    ("steps to MSE 0.002", lambda c, s: _t_mse(c, 0.002)),
    ("steps to 95 % cov.", lambda c, s: _scalar(c, "t95")),
    ("mean altitude", lambda c, s: _scalar(c, "altitude")),
    ("CPU s / step", lambda c, s: _scalar(c, "cpu_per_step")),
)


def _scalar(c: dict, key: str) -> np.ndarray:
    return np.array([np.nan if r["scalars"][key] is None else r["scalars"][key] for r in c["runs"]], float)


def _window(c: dict, lo: int, hi: int | None) -> np.ndarray:
    return np.array([np.mean(r["curves"]["mse"][lo:hi]) for r in c["runs"]])


def _t_mse(c: dict, level: float) -> np.ndarray:
    """Per run: first step with MSE <= level (NaN: never within the mission)."""
    out = []
    for r in c["runs"]:
        hit = np.flatnonzero(np.asarray(r["curves"]["mse"]) <= level)
        out.append(float(hit[0]) if hit.size else np.nan)
    return np.array(out)


def ci95(d: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half-width of the 95 % CI of the mean over runs: 1.96 s.e. (NaN-aware: runs that never reach
    a level are left out)."""
    n = np.sum(~np.isnan(d), axis=axis)
    return 1.96 * np.nanstd(d, axis=axis, ddof=1) / np.sqrt(n)


def split_step(ref: dict) -> int:
    t95 = _scalar(ref, "t95")
    return round(float(np.nanmean(t95))) if np.any(~np.isnan(t95)) else ref["steps"] // 2


def merge(results: list[dict]) -> dict:
    """Several results of the same runs as one (conditions of later results appended, duplicates
    of a condition id dropped): planners run in separate folders can be paired. Every condition
    must cover the same run seeds and steps as the others of its team setting."""
    out = {**results[0], "conditions": list(results[0]["conditions"])}
    ids = {c["id"] for c in out["conditions"]}
    for res in results[1:]:
        for c in res["conditions"]:
            if c["id"] not in ids:
                out["conditions"].append(c)
                ids.add(c["id"])
    for g, kc in groups(out).items():
        seeds = {(tuple(r["run_seed"] for r in c["runs"]), c["steps"]) for _, c in kc}
        if len(seeds) > 1:
            raise ValueError(f"{title(out, g)}: conditions on different runs or steps, cannot pair them")
    return out


def compare(res: dict, split: int | None = None, ref: str | None = None) -> list[dict]:
    """Per team setting: setting title, reference condition (planner `ref`, default the first),
    the other conditions, split step, stall threshold."""
    out = []
    for g, kc in groups(res).items():
        cs = [c for _, c in kc]
        r = next((c for c in cs if c["planner"] == ref), None) if ref else cs[0]
        if r is None:
            print(f"{title(res, g)}: no planner {ref}, skipped")
            continue
        out.append({"title": title(res, g), "ref": r, "others": [c for c in cs if c is not r],
                    "split": split if split is not None else split_step(r), "trap": res["meta"]["trap"]})
    return out


def primary(blocks: list[dict]) -> list[dict]:
    """The primary metrics per setting and planner (reference first): mean +- 95 % CI over runs, and
    for the others the paired difference to the reference (% of its mean +- 95 % CI)."""
    out = []
    for b in blocks:
        ref = b["ref"]
        for c in [ref, *b["others"]]:
            row = {"setting": b["title"], "planner": c["planner"], "ref": ref["planner"], "runs": len(c["runs"])}
            for name, f in (("avg_mse", lambda c: _scalar(c, "avg_mse")), ("t_mse_0.01", lambda c: _t_mse(c, 0.01)),
                            ("t_mse_0.002", lambda c: _t_mse(c, 0.002)),
                            ("cpu_per_step", lambda c: _scalar(c, "cpu_per_step"))):
                x, y = f(c), f(ref)
                d = x - y
                row[name], row[f"{name}_ci"] = np.nanmean(x), ci95(x)
                row[f"{name}_reached"] = int(np.sum(~np.isnan(x)))
                row[f"d_{name}_pct"] = 100 * np.nanmean(d) / np.nanmean(y) if c is not ref else 0.0
                row[f"d_{name}_ci"] = 100 * ci95(d) / abs(np.nanmean(y)) if c is not ref else 0.0
            stall = _scalar(c, "stall")
            row["stalled"] = float(np.mean(stall >= b["trap"]))
            row["worst_avg_mse"] = float(np.max(_scalar(c, "avg_mse")))
            out.append(row)
    return out


def _cell(r: dict, k: str, f: str = "{:.4f}") -> str:
    reach = "" if r[f"{k}_reached"] == r["runs"] else f" [{r[f'{k}_reached']}]"
    return (f"{f.format(r[k])} +- {f.format(r[k + '_ci'])} ({r['d_' + k + '_pct']:+5.1f} +- "
            f"{r['d_' + k + '_ci']:4.1f}){reach}")


def primary_table(rs: list[dict]) -> str:
    w = max(len(r["planner"]) for r in rs)
    head = (f"{'setting':32s} {'planner':{w}s} {'runs':>4s}  {'avg MSE (vs ref %)':>32s}  "
            f"{'steps to MSE 0.01 (vs ref %)':>34s}  {'steps to MSE 0.002 (vs ref %)':>36s}  {'stalled':>7s}"
            f" {'worst avg MSE':>13s}  {'CPU s/step':>10s}")
    lines = [head]
    for r in rs:
        lines.append(f"{r['setting'][:32]:32s} {r['planner']:{w}s} {r['runs']:4d}  {_cell(r, 'avg_mse'):>32s}  "
                     f"{_cell(r, 't_mse_0.01', '{:.1f}'):>34s}  {_cell(r, 't_mse_0.002', '{:.1f}'):>36s}  "
                     f"{r['stalled']:7.2f} {r['worst_avg_mse']:13.4f}  {r['cpu_per_step']:10.3f}")
    return "\n".join(lines) + ("\n[k] = runs that reached the level (steps averaged over those;"
                               " % over the runs where both did)")


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
    w = max(len(r["planner"]) for r in rs)
    head = (f"{'setting':34s} {'planner':{w}s} {'metric':20s} {'diff':>10s} {'% of ref':>18s}"
            f" {'lower/ties/runs':>16s} {'p (Wilcoxon)':>12s}")
    lines = [head]
    for r in rs:
        lines.append(f"{r['setting'][:34]:34s} {r['planner']:{w}s} {r['metric']:20s} {r['d']:+10.5f}"
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
    fig.suptitle("Paired difference to the reference per step (same runs), mean +- 95 % CI over runs\n"
                 "below 0 = lower than the reference; dashed = reference's mean step to 95 % coverage",
                 fontsize=9)
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
    fig.suptitle("Planner vs reference on the same runs: mean +- 95 % CI over runs\n"
                 "filled: Wilcoxon p < 0.05; < 0 = lower (better for MSE, steps and CPU)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
