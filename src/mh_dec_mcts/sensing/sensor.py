"""Sensor model: per-altitude error rates and observations.

Error rates are indexed by altitude level k:
    s0[k] = P(z=1 | m=0)   false positive
    s1[k] = P(z=0 | m=1)   false negative
They are computed ONCE per experiment (make_scenario) and reused by every run and step:
  - theoretical: closed form, instant.
  - empirical:   classifier predictions vs. annotations. Stored inside predictions.npz by
                 scripts/build_predictions.py; load_predictions reads the file once per process
                 and recomputes the rates only if they are missing.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

EPS = 1e-3  # keeps rates away from 0/1 so the Bayes update never divides by zero (legacy clip)


@dataclass(frozen=True)
class SensorRates:
    s0: np.ndarray  # (n_alt,)
    s1: np.ndarray  # (n_alt,)

    def at(self, k: int) -> tuple[float, float]:
        return float(self.s0[k]), float(self.s1[k])


# --------------------------------------------------------------------------- rates
def sigma(h, a: float = 1.0, b: float = 0.015):
    """Paper Eq. 1: a(1 - exp(-b h)), capped at 0.5 (a=1, b=0.015 never reaches the cap)."""
    return np.minimum(a * (1.0 - np.exp(-b * np.asarray(h, dtype=float))), 0.5)


def theoretical_rates(altitudes, a: float = 1.0, b: float = 0.015) -> SensorRates:
    s = sigma(altitudes, a, b)
    return SensorRates(s0=s.copy(), s1=s.copy())


def empirical_rates(pred_maps: np.ndarray, truth: np.ndarray) -> SensorRates:
    """Confusion rates of the classifier per altitude, over the whole annotated map.

    pred_maps: (n_alt, rows, cols) predicted labels; truth: (rows, cols) annotations.
    """
    if pred_maps.shape[1:] != truth.shape:
        raise ValueError(f"prediction maps {pred_maps.shape[1:]} != truth {truth.shape}")
    neg, pos = truth == 0, truth == 1
    if not neg.any() or not pos.any():
        raise ValueError("truth must contain both classes to estimate error rates")
    s0 = (pred_maps[:, neg] == 1).mean(axis=1)
    s1 = (pred_maps[:, pos] == 0).mean(axis=1)
    return SensorRates(s0=np.clip(s0, EPS, 1 - EPS), s1=np.clip(s1, EPS, 1 - EPS))


# --------------------------------------------------------------------- predictions
@dataclass(frozen=True)
class Predictions:
    maps: np.ndarray  # (n_alt, rows, cols) uint8
    altitudes: np.ndarray  # (n_alt,)
    rates: SensorRates  # empirical, computed once


def save_predictions(path, maps, altitudes, truth=None, **meta) -> None:
    """Write predictions.npz; if truth is given, the empirical rates are stored with it."""
    data = {"maps": np.asarray(maps, dtype=np.uint8), "altitudes": np.asarray(altitudes, float)}
    if truth is not None:
        r = empirical_rates(data["maps"], truth)
        data |= {"s0": r.s0, "s1": r.s1, "truth_sum": np.int64(truth.sum())}
    np.savez_compressed(path, **data, **{k: np.asarray(v) for k, v in meta.items()})


_CACHE: dict[str, Predictions] = {}  # one entry per predictions file, shared by all runs


def load_predictions(path, altitudes, truth) -> Predictions:
    """Load predictions.npz once per process; later calls return the same object.

    Empirical rates come from the file (written by save_predictions). If they are missing,
    or were computed against a different truth map, they are computed here once and cached.
    """
    key = str(Path(path).resolve())
    if key not in _CACHE:
        with np.load(key) as f:
            maps, alts = f["maps"], f["altitudes"]
            ok = "s0" in f and int(f["truth_sum"]) == int(truth.sum())
            rates = SensorRates(s0=f["s0"], s1=f["s1"]) if ok else None
        if rates is None:
            warnings.warn(f"{path}: no stored rates for this truth map; computing once")
            rates = empirical_rates(maps, truth)
        _CACHE[key] = Predictions(maps=maps, altitudes=alts, rates=rates)
    p = _CACHE[key]
    if len(p.altitudes) != len(altitudes) or not np.allclose(p.altitudes, altitudes, atol=1e-6):
        raise ValueError(
            f"{path}: cached altitudes {np.round(p.altitudes, 2)} != lattice "
            f"{np.round(altitudes, 2)}; rebuild with scripts/build_predictions.py"
        )
    if p.maps.shape[1:] != truth.shape:
        raise ValueError(f"{path}: maps {p.maps.shape[1:]} != truth {truth.shape}")
    return p


# --------------------------------------------------------------------- observation
def observe(truth, bounds, s0: float, s1: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic observation of truth[bounds] (legacy Field.get_observations, bit-exact).

    Exactly one rng.random(shape) draw per call; a cell is read correctly if u <= 1 - s.
    """
    i0, i1, j0, j1 = bounds
    m = truth[i0:i1, j0:j1]
    u = rng.random(m.shape)
    z0 = np.where((u <= 1.0 - s0) & (m == 0), 0, 1)
    z1 = np.where((u <= 1.0 - s1) & (m == 1), 1, 0)
    return np.where(m == 0, z0, z1)


def observe_pred(pred_maps: np.ndarray, k: int, bounds) -> np.ndarray:
    """Classifier observation: the precomputed prediction at altitude level k."""
    i0, i1, j0, j1 = bounds
    return pred_maps[k, i0:i1, j0:j1]