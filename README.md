# mh-dec-mcts
Hierarchical decentralized MCTS for multi-UAV active sensing and field mapping

## Running experiments

Every experiment runs through one runner, `scripts/run.py`, and is saved in one format,
`results.json` (see [docs/results_format.md](docs/results_format.md)). Plot scripts read only that
format. `run.py` runs a grid of conditions, planners × team sizes × (one `--setting`, or
`--sharing` × `--ranges`), with the same fields, starts and sensor noise for every planner
(run i uses seed + i). It prints tables (mean ± std over runs, paired difference to the first
planner ± s.e.) and writes them to `results.txt`. Add `--save-poses` for the trajectory and
first-seen figures.

Planner specs: `greedy` (the setting's IG / IGd), `ig`, `igd`, `random`, `sweep`,
`decmcts[:key=value,...]`, `mhmcts[:key=value,...]` (MH-MCTS keys such as `tile`, `g2_weight`
go to the config's `mh` section, the others to `mcts`).

### Pierdicca et al. (baseline reproduction)

Fig. 3: one UAV, Bayes only, IG vs Random vs Sweep:

    python scripts/run.py configs/pierdicca_fig3.yaml --planners ig random sweep --save-poses
    python scripts/plot_fig3.py results/runs/pierdicca_fig3_r5_none

Fig. 4: IG with pairwise weights equal / biased / adaptive × cluster radius r = 0..5
(the same loop is in `run.sh`):

    for w in equal biased adaptive; do for r in 0 1 2 3 4 5; do
      python scripts/run.py configs/pierdicca_fig4.yaml --planners ig --weights $w --r $r --out results/fig4
    done; done
    python scripts/plot_fig4.py results/fig4

Fig. 5: IG / IGd × sharing none / BS / BM × range 5 / 16 × N = 1, 2, 4, 8:

    bash scripts/fig5.sh            # = run.py ... --sharing none BS BM --ranges 5 16 --n 1 2 4 8, then plot_fig5.py
    python scripts/plot_fig5.py results/multi/pierdicca_fig5 --planners ig igd

### Planner comparison

Greedy vs Dec-MCTS vs MH-MCTS in one team setting of the config (`settings:` gives sharing,
range and the greedy baseline):

    python scripts/run.py configs/swarm_rect.yaml --setting IGd_BM_R5 --n 4 --save-poses \
        --out results/myopia --tag rect_n4 \
        --planners greedy decmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy,tile=12.5
    python scripts/plot_mission.py results/myopia/rect_n4     # mission.png, runs.png, firstseen.png

`scripts/experiments.sh` runs this for several configs and both settings.

### Horizon sweep

Greedy vs Dec-MCTS with horizon 2, 4, 8, 16, 24 on the square and the rectangle, 1 and 4 UAVs:

    python scripts/horizon_sweep.py --jobs 8           # runs what is missing, then the figure + CSV
    python scripts/horizon_sweep.py --plot-only        # figure from existing results (old ones too)

### Tuning Dec-MCTS

One parameter at a time around the config's `mcts`, scored against greedy on paired runs:

    python scripts/tune_decmcts.py configs/swarm.yaml --setting IGd_BM_R5 --jobs 8 \
        --param gamma 0.95 0.99 0.999 --param horizon 2 3 4 6

## Tests

    pytest -q                                   # unit tests
    pytest -m regression tests/regression       # every experiment vs the stored baseline, bit-exact (minutes)
