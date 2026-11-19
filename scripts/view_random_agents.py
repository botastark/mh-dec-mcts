"""Legacy-viewer figure for agents starting at random waypoints and moving randomly.

Checks grid / lattice / footprint / observation indexing by eye before any planner is involved.

    python scripts/view_random_agents.py configs/paper.yaml                 # 4 agents, 30 steps
    python scripts/view_random_agents.py configs/pc_60x110.yaml --agents 2 --steps 50
    python scripts/view_random_agents.py configs/wheat.yaml --every 5       # a figure every 5 steps
"""

import argparse
from pathlib import Path

import numpy as np

from mh_dec_mcts.mapping.belief import bayes_update
from mh_dec_mcts.scenario import load_config, make_scenario
from mh_dec_mcts.viewer import plot_terrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--every", type=int, default=0, help="also save a figure every N steps")
    ap.add_argument("--out", default="results/viewer")
    args = ap.parse_args()

    cfg = load_config(args.config, agents={"n": args.agents, "start": "random"})
    sc = make_scenario(cfg, run=args.run)
    lat = sc.lattice
    out = Path(args.out) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)

    paths = [[p] for p in sc.starts]
    belief = np.full(sc.grid.shape, 0.5)  # one shared map: plain Bayes, no LBP
    for t in range(args.steps + 1):
        for path in paths:
            p = path[-1]
            bayes_update(belief, lat.footprint(p), sc.observe(p), *sc.rates.at(p.k))
        if (args.every and t % args.every == 0) or t == args.steps:
            f = out / f"step_{t:03d}.png"
            plot_terrain(f, lat, sc.truth, belief, paths,
                         title=f"{cfg['name']}  |  {args.agents} agents, random moves  |  step {t}")
            print(f"-> {f}")
        if t < args.steps:
            for path in paths:
                nxt = lat.move(path[-1], str(sc.plan_rng.choice(lat.admissible(path[-1]))))
                assert nxt is not None
                path.append(nxt)

    print(f"{cfg['name']}: starts {[lat.xyz(p[0])[:2] for p in paths]}, "
          f"final MSE {((belief - sc.truth) ** 2).mean():.4f}")


if __name__ == "__main__":
    main()
