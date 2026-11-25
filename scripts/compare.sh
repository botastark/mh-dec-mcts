#!/usr/bin/env bash
# Compare all planners on the same runs: greedy (the setting's IG / IGd), Dec-MCTS and MH-MCTS,
# per config x team setting x team size (scripts/run.py). Data only: the figures from it any time with
# scripts/plot.py (mission, firstseen, curves, uavs), without rerunning.
#
#   bash scripts/compare.sh                                   # swarm_rect, both settings, 1 and 4 UAVs
#   CONFIGS="swarm swarm_rect" SETTINGS="IGd_BM_R5" NS="4" bash scripts/compare.sh
#   PLANNERS="greedy decmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy,tile=12.5" \
#       STEPS=300 NS=1 bash scripts/compare.sh
#   RUNS=2 STEPS=20 bash scripts/compare.sh                   # quick look
#
# Output in $OUT (default results/compare): <config>_<setting>_n<N>/ with results.json, results.npz,
# results.txt (tables: mean +- std over runs, paired difference to greedy +- s.e.); the terminal
# output of each in $OUT/logs/. Then e.g.: python scripts/plot.py mission $OUT/<tag>
# Long: start it and leave it, e.g.
#   nohup bash scripts/compare.sh > compare.log 2>&1 &
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=${CONFIGS:-"swarm_rect"}
SETTINGS=${SETTINGS:-"IGd_BM_R5 IG_BS_Rinf"}
NS=${NS:-"1 4"}
PLANNERS=${PLANNERS:-"greedy decmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy"}
JOBS=${JOBS:-$(nproc)}
OUT=${OUT:-results/compare}
RUNS=${RUNS:-}
STEPS=${STEPS:-}
extra=()
[[ -n "$RUNS" ]] && extra+=(--runs "$RUNS")
[[ -n "$STEPS" ]] && extra+=(--steps "$STEPS")

mkdir -p "$OUT/logs"
for c in $CONFIGS; do
  for s in $SETTINGS; do
    for n in $NS; do
      tag=${c}_${s}_n${n}
      log=$OUT/logs/$tag.txt
      echo "$(date '+%F %T') start $tag -> $log"
      python scripts/run.py "configs/$c.yaml" --setting "$s" --n "$n" --jobs "$JOBS" \
          --out "$OUT" --tag "$tag" --planners $PLANNERS ${extra[@]+"${extra[@]}"} > "$log" 2>&1 \
        && echo "$(date '+%F %T') done  $tag" || echo "$(date '+%F %T') FAILED $tag (see $log)"
    done
  done
done
