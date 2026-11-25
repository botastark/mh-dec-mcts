"""What the figure scripts plot, from results.json (experiment.load_results): mean and std over
runs per step (Pierdicca's convention), condition lookup, and planner colours."""

import numpy as np

from mh_dec_mcts.experiment import curve_stats

# Pierdicca's names -> curve names in results.json
PAPER = {"entropy": "entropy", "mse": "mse", "altitude": "alt", "coverage": "coverage_paper",
         "misalignment": "misalignment"}
PAPER_COLORS = {"ig": "green", "random": "blue", "sweep": "orange"}
PAPER_NAMES = {"ig": "IG", "igd": "IG$_d$", "random": "Random", "sweep": "Sweep", "decmcts": "Dec-MCTS",
               "mhmcts": "MH-MCTS"}


def band(cond: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std (ddof 0) over the runs of a condition's curve `key`. For old results without
    per-run curves: the stored mean and std curves (std 0 if none was stored)."""
    runs = cond["runs"]
    if runs and runs[0]["curves"] is not None:
        return curve_stats([r["curves"][key] for r in runs])
    mu = np.array(cond["mean_curves"][key])
    std = (cond.get("std_curves") or {}).get(key)
    return mu, np.array(std) if std is not None else np.zeros_like(mu)


def find(results: dict, **match) -> list[tuple[int, dict]]:
    """(index, condition) of the conditions whose fields equal `match` (planner, n, sharing, ...)."""
    return [(k, c) for k, c in enumerate(results["conditions"]) if all(c.get(f) == v for f, v in match.items())]


def label(spec: str) -> str:
    name, _, rest = spec.partition(":")
    return PAPER_NAMES.get(name, name) + (f" ({rest})" if rest else "")


OTHER = ["C3", "C4", "C5", "C6", "C7", "C8", "C9", "C0", "C1", "C2"]  # red, purple, ... before tab blue/orange/green


def colors(specs: list[str]) -> dict[str, str]:
    """Pierdicca's colours for IG / Random / Sweep, tab10 (red first) for every other planner."""
    out, k = {}, 0
    for s in specs:
        if s in PAPER_COLORS:
            out[s] = PAPER_COLORS[s]
        else:
            out[s] = OTHER[k % len(OTHER)]
            k += 1
    return out
