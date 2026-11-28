#!/usr/bin/env bash
# All runs for the report: rectangle, 100 steps, 20 runs per seed set; MCTS H=4, greedy rollouts,
# stride 1 unless the spec says otherwise. One folder per setting and seed set (paired inside).
# Resumable: rerun this script after a crash; finished runs are kept in <folder>/_partial/.
#   SMOKE=1 bash results/compare/logs/report.sh   # every spec for 1 run x 2 steps, into the scratchpad
cd /home/bota/repos/mh-dec-mcts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=configs/swarm_rect.yaml; L=${SMOKE_LOGS:-results/compare/logs}; OUT=results/compare; JOBS=16; N=(); EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then OUT=$SMOKE_OUT; JOBS=16; EXTRA=(--runs 1 --steps 2); fi
D="decmcts:horizon=4,rollout=greedy"; M="mhmcts:horizon=4,rollout=greedy"
STRIDES="greedy greedy:stride=8 $D $D,stride=2 $D,stride=4 $D,stride=8 $D,stride=8,stride_fine=1,stride_cov=0.9 $M $M,stride=8"
run() { local tag=$1; shift; [[ -f $OUT/$tag/results.json ]] && { echo "$(date '+%F %T') have  $tag"; return; }
  echo "$(date '+%F %T') start $tag"
  python scripts/run.py "$@" --jobs $JOBS --out $OUT --tag $tag "${EXTRA[@]}" > $L/$tag.txt 2>&1 \
    && echo "$(date '+%F %T') done  $tag" || echo "$(date '+%F %T') FAILED $tag"; }
laneA() {
  run report_n4_R5_H16_seed42_stride1     $C --setting IGd_BM_R5  --n 4 --runs 10 --planners greedy decmcts:horizon=16,rollout=greedy
  run report_n4_R5_seed42_strides         $C --setting IGd_BM_R5  --n 4 --planners $STRIDES
  run report_n4_R5_horizon_seed42_stride1 $C --setting IGd_BM_R5  --n 4 --planners greedy \
      decmcts:horizon=1,rollout=greedy decmcts:horizon=2,rollout=greedy decmcts:horizon=8,rollout=greedy
}
laneB() {
  run report_n4_Rinf_seed42_stride1       $C --setting IG_BS_Rinf --n 4 --planners greedy $D $M,g2_weight=0.5 $M $M,hl_rollouts=0
  run report_n1_seed42_strides            $C --setting IGd_BM_R5  --n 1 --planners $STRIDES
  run report_n4_R5_seed1042_strides       $C --setting IGd_BM_R5  --n 4 --seed 1042 --planners greedy greedy:stride=8 $D $D,stride=8 $M
  run report_n4_Rinf_seed1042_stride1     $C --setting IG_BS_Rinf --n 4 --seed 1042 --planners greedy $D $M
  run report_n1_seed1042_stride1          $C --setting IGd_BM_R5  --n 1 --seed 1042 --planners greedy $D $M
}
if [[ -n "${SMOKE:-}" ]]; then laneA; laneB; else laneA & laneB & wait; fi
echo "$(date '+%F %T') all done"
