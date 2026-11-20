"""News beliefs: what an agent has observed since it last shared (paper Sec. III-E).

    BS (broadcast single): one news belief per agent. After the agent's own observation it is
        updated (Bayes + one LBP step), fused into every neighbour's map and reset to 0.5 if
        it had at least one neighbour.
    BM (broadcast multi): one news belief per (agent, recipient) pair. Every one of them is
        updated with the agent's observation; each is fused into its recipient's map and reset
        only when that recipient is a neighbour. Nothing is lost while out of range.

Matches Pierdicca's _update_news_belief_LBP_and_fuse_single / _multi (news_inference_type
"LBP_single" / "LBP_multi"), including the order: all news updates first, then fusion agent by
agent, each fusion applied over the whole map.
"""

import numpy as np

from mh_dec_mcts.mapping.belief import bayes_update, lbp_update


def fuse(news: np.ndarray, belief: np.ndarray) -> None:
    """In place: belief <- normalised product with news (independent evidence), whole map."""
    mul = news * belief
    belief[:] = mul / (mul + (1.0 - news) * (1.0 - belief))


class News:
    def __init__(self, sharing: str, n_agents: int, shape: tuple[int, int]):
        if sharing not in ("none", "BS", "BM"):
            raise ValueError(f"unknown sharing '{sharing}'")
        self.sharing, self.n = sharing, n_agents
        if sharing == "BS":
            self.news = [np.full(shape, 0.5) for _ in range(n_agents)]
        elif sharing == "BM":  # news[a][k] = what a has for k (k == a unused)
            self.news = [[np.full(shape, 0.5) for _ in range(n_agents)] for _ in range(n_agents)]

    def step(self, beliefs: list[np.ndarray], bounds: list, zs: list, rates: list, psis: list,
             nbrs: list[list[int]]) -> None:
        """One round after every agent updated its own belief.

        bounds / zs / rates / psis: per agent footprint, observation, (s0, s1), pairwise potential
        (psi None = no LBP step, weights "none").
        nbrs: per agent neighbour ids at the current positions.
        """
        if self.sharing == "none":
            return
        if self.sharing == "BS":
            for a in range(self.n):
                bayes_update(self.news[a], bounds[a], zs[a], *rates[a])
            for a in range(self.n):
                if psis[a] is not None:
                    lbp_update(self.news[a], bounds[a], psis[a])
            for a in range(self.n):
                for nb in nbrs[a]:
                    fuse(self.news[a], beliefs[nb])
                if nbrs[a]:
                    self.news[a][:] = 0.5
        else:  # BM
            for a in range(self.n):
                for k in range(self.n):
                    if k != a:
                        bayes_update(self.news[a][k], bounds[a], zs[a], *rates[a])
            for a in range(self.n):
                for k in range(self.n):
                    if k != a and psis[a] is not None:
                        lbp_update(self.news[a][k], bounds[a], psis[a])
            for a in range(self.n):
                for nb in nbrs[a]:
                    fuse(self.news[a][nb], beliefs[nb])
                    self.news[a][nb][:] = 0.5
