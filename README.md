# mh-dec-mcts
Hierarchical decentralized MCTS for multi-UAV active sensing and field mapping

## Running experiments

Every experiment runs through one runner, `scripts/run.py`, and is saved in one format,
`results.json` (see [docs/results_format.md](docs/results_format.md)). `run.py` runs a grid of
conditions, planners × team sizes × (one `--setting`, or `--sharing` × `--ranges`), with the
same fields, starts and sensor noise for every planner (run i uses seed + i). It prints tables
(mean ± std over runs, paired difference to the first planner ± s.e.) and writes them to
`results.txt`.

Runs save data only: `results.json` (per-run curves and scalars) and `results.npz` (poses, each
UAV's own curves, first-seen maps, ground truth; `--save-beliefs` adds the final belief maps).
Every figure is drawn afterwards from these files with `scripts/plot.py`, into `figures/` next to
the data, so a plot can be changed or added without rerunning anything.

Planner specs: `greedy` (the setting's IG / IGd), `ig`, `igd`, `random`, `sweep`,
`decmcts[:key=value,...]`, `mhmcts[:key=value,...]` (MH-MCTS keys such as `tile`, `g2_weight`,
`hl_rollouts` go to the config's `mh` section, the others to `mcts`).

| script | what for |
|---|---|
| `scripts/pierdicca.sh` | Pierdicca et al. replica, Figs. 3, 4, 5: checks mapping, baseline planners and comms |
| `scripts/tune.py` | tune Dec-MCTS or MH-MCTS against greedy (one parameter at a time; also the horizon sweep) |
| `scripts/compare.sh` | compare all planners (greedy, Dec-MCTS, MH-MCTS) per config × setting × team size |
| `scripts/run.py` | the runner behind all three |
| `scripts/plot.py` | every figure, from saved results (`mission`, `firstseen`, `curves`, `uavs`, `fig3`, `fig4`, `fig5`, `tune`) |

### Pierdicca et al. replica (Figs. 3, 4, 5)

The baseline every planner is compared against, and the check that mapping (Bayes + LBP), IG /
IGd / Random / Sweep and the comms (news sharing none / BS / BM, comms range, team size) behave
as in the paper. `compat: pierdicca` configs reproduce his random streams.

    bash scripts/pierdicca.sh                    # runs for all three figures -> results/pierdicca/fig{3,4,5}/
    FIGS="5" bash scripts/pierdicca.sh           # one figure's runs
    RUNS=2 STEPS=20 bash scripts/pierdicca.sh    # quick look
    python scripts/plot.py fig3 results/pierdicca/fig3      # -> results/pierdicca/fig3/figures/paper_fig3.png
    python scripts/plot.py fig4 results/pierdicca/fig4
    python scripts/plot.py fig5 results/pierdicca/fig5

One figure by hand, e.g. Fig. 5:

    python scripts/run.py configs/pierdicca_fig5.yaml --planners ig igd --sharing none BS BM \
        --ranges 5 16 --n 1 2 4 8 --out results/pierdicca --tag fig5
    python scripts/plot.py fig5 results/pierdicca/fig5

### Tuning Dec-MCTS and MH-MCTS

One parameter at a time around a base, every candidate scored against greedy on the same runs
(seed 1000 by default, so you tune on other runs than you report):

    python scripts/tune.py configs/swarm.yaml --setting IGd_BM_R5 \
        --param gamma 0.95 0.99 0.999 --param cp 0.25 0.5 1.0 --param horizon 2 3 4 6
    python scripts/tune.py configs/swarm_rect.yaml --planner mhmcts --base horizon=4,rollout=greedy \
        --n 1 --steps 300 --param tile 6.25 12.5 25 --param hl_rollouts 0 10

The horizon sweep (is there one good Dec-MCTS horizon?) is a tuning of `horizon`, one call per
field and team size, then one figure of all of them:

    python scripts/tune.py configs/swarm.yaml --base rollout=greedy --param horizon 2 4 8 16 24 \
        --n 1 --steps 300 --seed 42 --tag square_n1
    python scripts/tune.py configs/swarm.yaml --base rollout=greedy --param horizon 2 4 8 16 24 \
        --n 4 --steps 100 --seed 42 --tag square_n4
    # ... rect_n1, rect_n4 with configs/swarm_rect.yaml, then:
    python scripts/plot.py tune results/tune/square_n1 results/tune/square_n4 \
        results/tune/rect_n1 results/tune/rect_n4 --out paper/figures

### Comparing all planners

    bash scripts/compare.sh                      # swarm_rect, both settings, 1 and 4 UAVs -> results/compare/
    CONFIGS="swarm swarm_rect" NS="4" PLANNERS="greedy decmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy,tile=12.5" \
        bash scripts/compare.sh

One comparison by hand, then figures from its data (as many and as often as needed):

    python scripts/run.py configs/swarm_rect.yaml --setting IGd_BM_R5 --n 4 --out results/compare \
        --tag rect_n4 --planners greedy decmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy
    python scripts/plot.py mission results/compare/rect_n4      # mission.png, runs.png, firstseen.png (worst run)
    python scripts/plot.py firstseen results/compare/rect_n4 --run 7
    python scripts/plot.py curves results/compare/rect_n4 --keys mse team_entropy coverage_paper
    python scripts/plot.py uavs results/compare/rect_n4 --planner greedy   # each UAV's own MSE / entropy / altitude

## Tests

    pytest -q                                   # unit tests
    pytest -m regression tests/regression       # every experiment vs the stored baseline, bit-exact (~1 min)
