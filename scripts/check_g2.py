"""Step-2 check: mission-time estimates on a scenario (unknown field, and west half known).

    python scripts/check_g2.py configs/swarm_rect.yaml [--tile 25] [--done 0.3] [--level 0]
"""

import argparse

import numpy as np

from mh_dec_mcts.planning.mh_mcts import MissionTime, Regions, look_rate
from mh_dec_mcts.planning.objective import PlanGrid
from mh_dec_mcts.scenario import load_config, make_scenario

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--tile", type=float, default=25.0)
ap.add_argument("--done", type=float, default=0.3)
ap.add_argument("--level", type=int, default=0, help="altitude level of the nominal look (rate)")
a = ap.parse_args()

sc = make_scenario(load_config(a.config), 0)
lat = sc.lattice
reg = Regions(lat, 8, a.tile)
lam = look_rate(lat, sc.rates, 8, a.level)
T = MissionTime(reg, lam, a.done)
unknown = PlanGrid(np.full(sc.grid.shape, 0.5), lat, sc.rates, 8).base
E = reg.entropy(unknown)
print(f"{reg.n} regions; rate {lam:.0f} bits/step (look at level {a.level}); "
      f"work per unknown region {T.work(E, 0):.1f} steps; tile side {reg.per} steps")
sw, se, nw, ne = (0, 0), (lat.nx - 1, 0), (0, lat.ny - 1), (lat.nx - 1, lat.ny - 1)
op = T.open(unknown)
print(f"unknown field, 1 UAV at SW corner : T = {T(E, op, [(0, sw)]):6.1f} steps")
print(f"unknown field, 4 UAVs at corners  : T = {T(E, op, [(0, p) for p in (sw, se, nw, ne)]):6.1f} steps")
half = unknown.copy()
half[:, : half.shape[1] // 2] = 0.999  # west half known
Eh, oh = reg.entropy(half), T.open(half)
print(f"west half known: open regions {sorted(oh)}")
for name, xy in [("west edge", (0, lat.ny // 2)), ("middle", (lat.nx // 2, lat.ny // 2)),
                 ("east edge", (lat.nx - 1, lat.ny // 2))]:
    print(f"  1 UAV at the {name:9s} {xy}: T = {T(Eh, oh, [(0, xy)]):6.1f} steps")
