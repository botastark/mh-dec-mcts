"""Draw the MH-MCTS regions over a scenario's field and lattice (check for step 1).

    python scripts/show_regions.py configs/swarm_rect.yaml [--tile 25] [--run 0] [--out FILE]

Default output: results/regions/regions_<config>_tile<tile>m_<nrx>x<nry>_run<run>.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mh_dec_mcts.planning.mh_mcts import Regions
from mh_dec_mcts.scenario import load_config, make_scenario

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--tile", type=float, default=25.0)
ap.add_argument("--run", type=int, default=0, help="run index: field of seed + run")
ap.add_argument("--out", default=None, help="output file (default: name with the settings)")
a = ap.parse_args()

cfg = load_config(a.config)
sc = make_scenario(cfg, a.run)
lat, g = sc.lattice, sc.grid
reg = Regions(lat, 8, a.tile)
x0, x1 = g.x_range
y0, y1 = g.y_range
fig, ax = plt.subplots(figsize=(10, 10 * g.size_y / g.size_x + 0.8))
ax.imshow(sc.truth, cmap="Greys", alpha=0.35, extent=(x0, x1, y0, y1), interpolation="nearest")
wx = lat.x0 + np.arange(lat.nx) * lat.xy_step
wy = lat.y0 + np.arange(lat.ny) * lat.xy_step
ax.plot(*np.meshgrid(wx, wy), "k.", ms=2)
edge = reg.per * lat.xy_step  # tile edges in metres; a waypoint on an edge belongs to both tiles
for k in range(1, reg.nrx):
    ax.axvline(lat.x0 + k * edge, color="C3", lw=1.2)
for k in range(1, reg.nry):
    ax.axhline(lat.y0 + k * edge, color="C3", lw=1.2)
for r in range(reg.n):
    cx, cy = reg.center[r]
    ax.text(wx[cx], wy[cy], str(r), color="C3", fontsize=14, ha="center", va="center", weight="bold")
ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x (m)", ylabel="y (m)",
       title=f"{cfg['name']}, run {a.run} (seed {sc.run_seed}): {reg.n} regions ({reg.nrx} x {reg.nry}), {reg.per} waypoint steps = "
             f"{reg.per * lat.xy_step:g} m per side; lattice {lat.nx} x {lat.ny}")
ax.set_aspect("equal")
fig.tight_layout()
out = Path(a.out or f"results/regions/regions_{cfg['name']}_tile{a.tile:g}m_{reg.nrx}x{reg.nry}_run{a.run}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print(f"saved {out}: {reg.n} regions, cells per region (sub-grid) {reg.cells.astype(int).tolist()}")
