"""Belief update for one agent: Bayes (occupancy grid) + one step of loopy BP.

Matches legacy mapper_LBP.OccupancyMap as used by the simulator:
    update_belief_OG(...)  then  propagate_messages(..., max_iterations=1, reset_msgs=True)
Because messages are reset to 0.5 every step and only one iteration runs, LBP is a pure
function of the belief inside the footprint: no message state is kept between steps.

belief[i, j] = P(m_ij = 1). Footprint bounds are half-open (i0, i1, j0, j1).
"""

import numpy as np

PSI_EQUAL = np.array([[0.5, 0.5], [0.5, 0.5]])
PSI_BIASED = np.array([[0.7, 0.3], [0.3, 0.7]])


def bayes_update(belief: np.ndarray, bounds, z: np.ndarray, s0: float, s1: float) -> None:
    """In place, paper Eq. 2. s0 = P(z=1|m=0), s1 = P(z=0|m=1)."""
    i0, i1, j0, j1 = bounds
    prior = belief[i0:i1, j0:j1]
    l0 = np.where(z == 0, 1 - s0, s0)  # P(z | m=0)
    l1 = np.where(z == 0, s1, 1 - s1)  # P(z | m=1)
    p0 = l0 * (1.0 - prior)
    p1 = l1 * prior
    belief[i0:i1, j0:j1] = p1 / (p0 + p1)


def adaptive_psi(z: np.ndarray) -> np.ndarray:
    """Paper Sec. III-C: Pearson correlation between centre value and number of occupied
    4-neighbours over non-overlapping 3x3 windows of the observation, squashed by a sigmoid."""
    pr, pc = (-z.shape[0]) % 3, (-z.shape[1]) % 3
    o = np.pad(z, ((0, pr), (0, pc)), mode="edge") if pr or pc else z
    br, bc = o.shape[0] // 3, o.shape[1] // 3
    v = o.reshape(br, 3, bc, 3).swapaxes(1, 2).reshape(br * bc, 3, 3)
    centre = v[:, 1, 1].astype(float)
    n_occ = np.count_nonzero(np.stack([v[:, 0, 1], v[:, 1, 2], v[:, 2, 1], v[:, 1, 0]], axis=1), axis=1).astype(float)
    a = centre - centre.mean()
    b = n_occ - n_occ.mean()
    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    rho = np.clip(np.sum(a * b) / max(den, 1e-12), -1.0, 1.0)
    w = 1 / (1 + np.exp(-rho))
    return np.array([[w, 1 - w], [1 - w, w]])


def pairwise_psi(kind: str, z: np.ndarray) -> np.ndarray:
    if kind == "equal":
        return PSI_EQUAL
    if kind == "biased":
        return PSI_BIASED
    if kind == "adaptive":
        return adaptive_psi(z)
    raise ValueError(f"unknown pairwise weights '{kind}'")


def lbp_update(belief: np.ndarray, bounds, psi: np.ndarray) -> None:
    """In place: one LBP iteration over the footprint, messages starting at 0.5.

    Each cell receives one message from each 4-neighbour inside the footprint (0.5 from
    outside / off-grid). The final belief is the normalised product of the four messages and
    the cell's own belief, multiplied in the legacy channel order (above, right, below, left,
    self) so results are bit-identical.
    """
    i0, i1, j0, j1 = bounds
    b = belief[i0:i1, j0:j1]
    # message a cell sends to a neighbour: its own belief times three 0.5 messages, through psi
    mul0 = 0.5 * 0.5 * 0.5 * (1 - b)
    mul1 = 0.5 * 0.5 * 0.5 * b
    m0 = psi[0, 0] * mul0 + psi[0, 1] * mul1
    m1 = psi[1, 0] * mul0 + psi[1, 1] * mul1
    m = m1 / (m0 + m1)

    from_above = np.full_like(b, 0.5)
    from_above[1:, :] = m[:-1, :]
    from_right = np.full_like(b, 0.5)
    from_right[:, :-1] = m[:, 1:]
    from_below = np.full_like(b, 0.5)
    from_below[:-1, :] = m[1:, :]
    from_left = np.full_like(b, 0.5)
    from_left[:, 1:] = m[:, :-1]

    bel1 = from_above * from_right * from_below * from_left * b
    bel0 = (1 - from_above) * (1 - from_right) * (1 - from_below) * (1 - from_left) * (1 - b)
    belief[i0:i1, j0:j1] = bel1 / (bel0 + bel1)


def update(belief: np.ndarray, bounds, z: np.ndarray, s0: float, s1: float, weights: str) -> None:
    """One observation: Bayes update, then LBP with the chosen pairwise weights (legacy order)."""
    bayes_update(belief, bounds, z, s0, s1)
    lbp_update(belief, bounds, pairwise_psi(weights, z))