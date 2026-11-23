# Result format (`results.json`, schema 1)

Every experiment run by `scripts/run.py` is saved in one folder, `<out>/<tag>/`:

| file | content |
|---|---|
| `results.json` | every run's per-step curves and per-run scalars (below) |
| `results.npz` | only with `--save-poses`: poses, first-seen maps and ground truth per run |
| `results.txt` | the tables `run.py` printed |

Only per-run data is stored. Means, std and s.e. are computed when plotting or tabulating
(`mh_dec_mcts.experiment.curve_stats`, `se`, `paired_diff`). Read with
`mh_dec_mcts.experiment.load_results(folder)` and `load_arrays(folder)`. JSON floats round-trip
exactly; nothing is rounded.

```
{
  "schema": 1,
  "meta": {
    "config": "<config path as given>",
    "config_resolved": {...},          # load_config + command-line overrides (weights, r, runs, ...)
    "git_commit": "<hash>" | null,
    "created": "<ISO time>",
    "command": ["scripts/run.py", ...],
    "runs": R, "steps": T,
    "late": 10, "stall_rate": 0.01, "trap": 15,   # backfill / stall parameters
    "setting": "<name>" | null,
    "compat": null | "pierdicca"
  },
  "conditions": [
    {
      "id": "<planner spec>|n=<n>|sharing=<s>|range=<R>|setting=<name or None>",
      "planner": "<spec>",              # e.g. greedy, ig, decmcts:horizon=4,rollout=greedy
      "n": <UAVs>, "sharing": "none" | "BS" | "BM", "range": <int> | null,
      "setting": "<name>" | null,
      "greedy": "ig" | "igd" | null,    # what the spec "greedy" means here (from the setting)
      "steps": T,
      "runs": [
        {"run": i, "run_seed": <seed + i>,
         "curves": {<name>: [T floats]},
         "scalars": {<name>: number}}
      ]
    }
  ]
}
```

## Curves (one value per step, `steps` values)

Step t is recorded before the t-th move (t = 0: the prior), except where noted. Per-UAV
quantities are averaged over the UAVs of the run (Pierdicca's convention).

| name | meaning |
|---|---|
| `mse` | mean over cells of (truth - belief)^2, mean over the UAVs' own maps |
| `entropy` | sum of cell entropies (bits), mean over the UAVs' own maps |
| `alt` | flight altitude (m), mean over the UAVs |
| `misalignment` | variance across the UAVs' maps, mean over cells (`sim.metrics.misalignment`) |
| `coverage_paper` | Pierdicca's coverage: fraction of the field inside a footprint of any UAV at any pose up to the current one (start included) |
| `coverage` | fraction of the field seen from the flown poses, after move t (start excluded) |
| `backfill` | cumulative fraction of the field first seen more than `late` steps after its x-column was first seen (after move t) |
| `overlap` | share of the ground under the UAVs that is under >= 2 of them (after move t) |
| `connected` | share of UAVs with a teammate in comms range (after move t) |
| `team_entropy` | entropy of one map that got every UAV's observations (recorded, not plotted) |
| `plan` | CPU seconds in the planner per step, whole team (timing: differs between identical runs) |

## Scalars (one value per run)

Names as in the old `runs.json` of `myopia.py`:
`avg_mse` (mean of `mse` over the steps), `final_mse`, `stall` (longest stretch of steps the
team's entropy falls by < `stall_rate` of what is left while > 20 % remains), `final_cov`,
`t80` / `t95` (first step with `coverage` >= 0.8 / 0.95, or null), `altitude` (mean over steps
and UAVs), `balance` (coefficient of variation of the cells each UAV saw first),
`cpu_per_step` (CPU seconds per step of the whole run). Also `plan_per_step` (mean of `plan`)
and `wall` (wall seconds of the run). The three timings differ between otherwise identical runs.

## `results.npz` (with `--save-poses`)

| key | array |
|---|---|
| `c<k>_r<i>_poses` | (UAVs, steps + 1, 3) int: (ix, iy, altitude level) per pose, start first; condition k, run i |
| `c<k>_r<i>_first` | (rows, cols) int32: step at which each cell was first seen, -1 = never |
| `truth_r<i>` | (rows, cols) uint8: ground truth of run i (the same for every condition: paired) |

## Old results

`load_results` also reads the `runs.json` of the old `myopia.py` / `horizon_sweep.py`. It returns
it in the same shape (`schema` 0), with `curves = None` for every run and the stored mean (and
std, if present) curves as `mean_curves` / `std_curves` of each condition. Summary plots and
tables that only need scalars work on both.
