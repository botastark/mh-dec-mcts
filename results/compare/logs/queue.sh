#!/usr/bin/env bash
# Stride study, then the comparisons rerun without the stride (stride 1 = full grid, as greedy).
# Seed 42 (config): every run pairs with the earlier results. Detached; progress in queue.log.
cd /home/bota/repos/mh-dec-mcts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/compare; L=$OUT/logs
D="decmcts:horizon=4,rollout=greedy"
run() { local tag=$1; shift; echo "$(date '+%F %T') start $tag"
  python scripts/run.py "$@" --jobs 28 --out $OUT --tag $tag > $L/$tag.txt 2>&1 \
    && echo "$(date '+%F %T') done  $tag" || echo "$(date '+%F %T') FAILED $tag"; }
run stride_study configs/swarm_rect.yaml --setting IGd_BM_R5 --n 1 4 --steps 100 --planners greedy \
  $D,stride=1 $D,stride=2 $D,stride=4 $D,stride=8 \
  $D,stride=8,stride_fine=1,stride_cov=0.5 $D,stride=8,stride_fine=1,stride_cov=0.9 $D,stride=4,stride_fine=1,stride_cov=0.9
run rect_n4_bm_s1 configs/swarm_rect.yaml --setting IGd_BM_R5 --n 4 --steps 100 --planners greedy \
  $D decmcts:horizon=8,rollout=greedy decmcts:horizon=16,rollout=greedy mhmcts:horizon=4,rollout=greedy
run rect_n4_rinf_s1 configs/swarm_rect.yaml --setting IG_BS_Rinf --n 4 --steps 100 --planners greedy \
  $D mhmcts:horizon=4,rollout=greedy,g2_weight=0.5 mhmcts:horizon=4,rollout=greedy mhmcts:horizon=4,rollout=greedy,hl_rollouts=0
echo "$(date '+%F %T') all done"
