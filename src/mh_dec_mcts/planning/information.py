"""Entropy and information gain (paper Eq. 3-8; legacy helper.H / cH / information_gain)."""

import numpy as np


def entropy(p: np.ndarray) -> np.ndarray:
    """Compute binary entropy of a random variable (or belief map)."""

    assert not np.any(np.isnan(p)), f"NaN detected in var: {p}"
    var = np.clip(p, 0.0, 1.0)  # Clamps values to the range [0, 1]

    v1 = var
    v2 = 1.0 - var

    if isinstance(var, np.ndarray):
        v1 = np.where(v1 == 0.0, 1.0, v1)
        v2 = np.where(v2 == 0.0, 1.0, v2)
    else:
        if v1 == 0.0:
            v1 = 1.0
        if v2 == 0.0:
            v2 = 1.0

    return -v1 * np.log2(v1) - v2 * np.log2(v2)


def binary_entropy(p):
    """Binary entropy in bits for the planner's inner loops: p clipped to [1e-12, 1 - 1e-12],
    no NaN check. entropy() is the legacy-exact version (exactly 0 at p = 0 or 1) behind the
    reported metrics."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def info_gain(prior: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """Compute information gain (paper Eq. 3-8).
    
    Per-cell expected IG = H(m) - H(m|z) for one observation with rates (s0, s1).

    Legacy details kept: rates rounded to 7 decimals; IG set to exactly 0 where H(m) and
    H(m|z) agree to np.isclose precision (removes float noise that would break ties).
    """
    s0 = float(np.clip(np.round(s0, 7), 0.0, 1.0))
    s1 = float(np.clip(np.round(s1, 7), 0.0, 1.0))
    pz0 = (1.0 - s0) * (1.0 - prior) + s1 * prior  # p(z=0)
    pz1 = 1.0 - pz0 # p(z=1)

    # posterior distribution probabilities
    # p(m = 1|z = 0) = (p(z = 0|m = 1)p(m = 1))/p(z = 0)
    prior = np.asarray(prior, dtype=float)
    p10 = np.divide(s1 * prior, pz0, out=prior.copy(), where=pz0 > 0)
    p11 = np.divide((1.0 - s1) * prior, pz1, out=prior.copy(), where=pz1 > 0)
    h = entropy(prior)
    ch = pz0 * entropy(p10) + pz1 * entropy(p11)
    return np.where(np.isclose(h, ch), 0.0, h - ch)


