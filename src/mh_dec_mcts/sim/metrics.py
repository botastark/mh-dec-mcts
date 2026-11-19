"""Map-quality metrics (paper Sec. IV). Shared by the single- and multi-agent loops."""

import numpy as np

from mh_dec_mcts.planning.information import entropy


def total_entropy(belief: np.ndarray) -> float:
    """Sum of cell entropies in bits (paper Eq. 4 summed over the field)."""
    return float(entropy(belief).sum())


def mse(belief: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((truth - belief) ** 2))


def coverage(seen: np.ndarray) -> float:
    """Fraction of cells inside at least one footprint so far."""
    return float(seen.mean())


# TODO multi-agent: belief misalignment (paper Fig. 5). Not defined in the paper or the
# legacy repo; take the definition from Pierdicca's Precision-Agriculture-Dev code.
