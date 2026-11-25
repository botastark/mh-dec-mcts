"""Pairwise weights of the belief update (mapping/belief.py)."""

import numpy as np

from mh_dec_mcts.mapping.belief import adaptive_psi, adaptive_psi_pierdicca


def test_pierdicca_weights_equal_when_the_correlation_is_undefined():
    """Every 3x3 window centre the same, neighbour counts differing: np.corrcoef gives NaN
    (Pierdicca's code stops there); the weights fall back to equal, as adaptive_psi."""
    z = np.ones((27, 27), dtype=int)
    z[0, 1] = 0  # one window loses its top neighbour; every centre (rows/cols 1, 4, 7, ...) stays 1
    with np.errstate(all="raise"):  # no warning either
        psi = adaptive_psi_pierdicca(z)
    np.testing.assert_array_equal(psi, [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_array_equal(adaptive_psi(z), psi)


def test_pierdicca_weights_unchanged_where_the_correlation_is_defined():
    rng = np.random.default_rng(0)
    for shape in [(25, 25), (24, 30), (30, 21)]:
        z = (rng.random(shape) < 0.5).astype(int)
        pr, pc = (-shape[0]) % 3, (-shape[1]) % 3
        o = np.pad(z, ((0, pr), (0, pc)), mode="edge")
        v = (o.reshape((max(o.shape[0] // 3, o.shape[1] // 3), 3, -1, 3)).swapaxes(1, 2)
             .reshape((o.shape[0] // 3) * (o.shape[1] // 3), 3, 3))
        n_occ = np.count_nonzero(np.stack([v[:, 0, 1], v[:, 1, 2], v[:, 2, 1], v[:, 1, 0]], axis=1), axis=1)
        w = 1 / (1 + np.exp(-np.corrcoef(v[:, 1, 1], n_occ)[0, 1]))  # his formula, unguarded
        assert adaptive_psi_pierdicca(z)[0, 0] == w
