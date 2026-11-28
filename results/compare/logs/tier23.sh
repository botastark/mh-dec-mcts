#!/usr/bin/env bash
# Tier 2 (mechanism) and Tier 3 (exploratory) runs: rectangle, 100 steps, seed 42 (Tier 3 is tuned
# here and tested once on 1042 later). Tree planners at stride 2 (as good as stride 1 at 1/4 of the
# cost, report_*_strides). Folders that add conditions to a report_* folder have 20 runs on the same
# seeds, so they pair with it:  python scripts/plot.py paired REPORT T23 --merge --ref SPEC
# The stress folders hold their own references (10 runs).
# Resumable: rerun after a crash; finished runs are kept in <folder>/_partial/.
#   SMOKE=1 SMOKE_OUT=... SMOKE_LOGS=... bash results/compare/logs/tier23.sh   # 1 run x 2 steps each
cd /home/bota/repos/mh-dec-mcts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=configs/swarm_rect.yaml; L=${SMOKE_LOGS:-results/compare/logs}; OUT=results/compare; JOBS=11; EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then OUT=$SMOKE_OUT; JOBS=16; EXTRA=(--runs 1 --steps 2); fi
D="decmcts:horizon=4,rollout=greedy,stride=2"; M="mhmcts:horizon=4,rollout=greedy,stride=2"
MHG="mhgreedy mhgreedy:budget=true mhgreedy:hl_rollouts=0 mhgreedy:budget=true,hl_rollouts=0"
run() { local tag=$1; shift; [[ -f $OUT/$tag/results.json ]] && { echo "$(date '+%F %T') have  $tag"; return; }
  echo "$(date '+%F %T') start $tag"
  python scripts/run.py "$@" --jobs $JOBS --out $OUT --tag $tag "${EXTRA[@]}" > $L/$tag.txt 2>&1 \
    && echo "$(date '+%F %T') done  $tag" || echo "$(date '+%F %T') FAILED $tag"; }
lane1() {  # tree runs, 4 UAVs, range 5: stride study (MH), root log, b(1-b) reward, planner seeds
  run t23_n4_R5_seed42_trees $C --setting IGd_BM_R5 --n 4 --planners \
      mhmcts:horizon=4,rollout=greedy,stride=2 mhmcts:horizon=4,rollout=greedy,stride=4 \
      $D,log_root=true $D,reward=variance $D,planner_seed=1 $D,planner_seed=2
}
lane2() {  # full comms and one UAV
  run t23_n4_Rinf_seed42 $C --setting IG_BS_Rinf --n 4 --planners \
      $D,log_root=true $D,reward=variance $M $M,switch_cov=0.9 $MHG
  run t23_n1_seed42 $C --setting IGd_BM_R5 --n 1 --planners \
      $M mhmcts:horizon=4,rollout=greedy,stride=4 greedy:stride=2 greedy:stride=4 \
      $D,log_root=true $D,reward=variance $MHG
}
lane3() {  # cheap planners, the greedy noise copy on both seed sets, stress scenarios
  run t23_n4_R5_seed42_cheap $C --setting IGd_BM_R5 --n 4 --planners \
      greedy:stride=2 greedy:stride=4 greedy:planner_seed=1 greedy:planner_seed=2 $MHG \
      $M,switch_cov=0.5 $M,switch_cov=0.9
  run t23_n4_R5_seed1042_greedyseeds $C --setting IGd_BM_R5 --n 4 --seed 1042 --planners \
      greedy:planner_seed=1 greedy:planner_seed=2
  run t23_stress_depot_n4_R5 $C --setting IGd_BM_R5 --n 4 --start depot --runs 10 --planners \
      greedy $D $M mhgreedy mhgreedy:budget=true
  run t23_stress_n4_R2 $C --setting IGd_BM_R2 --n 4 --runs 10 --planners greedy $D $M mhgreedy mhgreedy:budget=true
  run t23_stress_r3_n4_R5 $C --setting IGd_BM_R5 --n 4 --r 3 --runs 10 --planners greedy $D
  run t23_stress_r8_n4_R5 $C --setting IGd_BM_R5 --n 4 --r 8 --runs 10 --planners greedy $D
}
if [[ -n "${SMOKE:-}" ]]; then lane1; lane2; lane3; else lane1 & lane2 & lane3 & wait; fi
echo "$(date '+%F %T') all done"
