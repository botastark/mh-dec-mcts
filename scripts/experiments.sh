#!/usr/bin/env bash
# Myopia / team experiments (default: the 200 m x 100 m Gaussian field, ref 50 m) x both settings, 20 independent runs x 100 steps
# each (from the configs), greedy vs Dec-MCTS. Results in results/myopia/<config>_<setting>/,
# terminal tables in results/myopia/logs/. Long: start it and leave it, e.g.
#   nohup bash scripts/experiments.sh > experiments.log 2>&1 &
#   JOBS=16 bash scripts/experiments.sh                     # more parallel processes
#   CONFIGS="swarm_rect wheat_trap" bash scripts/experiments.sh
set -u
JOBS=${JOBS:-8}
CONFIGS=${CONFIGS:-"swarm_big"}   # also: swarm swarm_rect
SETTINGS=${SETTINGS:-"IGd_BM_R5 IG_BS_Rinf"}
PLANNERS=${PLANNERS:-"greedy decmcts:horizon=2 decmcts:horizon=8,rollout=greedy decmcts:horizon=16,rollout=greedy"}
export OMP_NUM_THREADS=1
mkdir -p results/myopia/logs
for c in $CONFIGS; do
  for s in $SETTINGS; do
    log=results/myopia/logs/${c}_${s}.txt
    echo "$(date '+%F %T') start $c $s -> $log"
    python scripts/myopia.py "configs/$c.yaml" --setting "$s" --jobs "$JOBS" --planners $PLANNERS > "$log" 2>&1 \
      && echo "$(date '+%F %T') done  $c $s" || echo "$(date '+%F %T') FAILED $c $s (see $log)"
  done
done
