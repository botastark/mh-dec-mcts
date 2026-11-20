"""Who can talk to whom (Pierdicca's Proximity).

Two agents are neighbours when they are within `range` lattice steps of each other along x,
along y AND along the altitude levels (a box, not a circle). range=None means unlimited.
His R = 5 is range=5 (15.6 m horizontally); his R = infinity is range=16, which covers the
whole 17 x 17 x 6 lattice, i.e. the same as None.
"""

from mh_dec_mcts.sensing.lattice import Pose


def neighbours(poses: list[Pose], range_: int | None) -> list[list[int]]:
    """For each agent, the ids of its neighbours (ascending, itself excluded)."""
    out = []
    for i, p in enumerate(poses):
        out.append([
            j for j, q in enumerate(poses)
            if j != i and (range_ is None or
                           (abs(p.ix - q.ix) <= range_ and abs(p.iy - q.iy) <= range_ and abs(p.k - q.k) <= range_))
        ])
    return out
