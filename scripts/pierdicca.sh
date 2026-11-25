#!/usr/bin/env bash
# Pierdicca et al. replica, Figs. 3, 4 and 5: the runs (scripts/run.py; data only, the figures from
# them any time with scripts/plot.py, printed at the end). Checks the mapping
# (Bayes + LBP, pairwise weights), the baseline planners (IG, IGd, Random, Sweep) and the comms
# (news sharing none / BS / BM, comms ranges, team sizes) against the paper. compat: pierdicca
# configs: his random streams, so the runs of a condition run in order (conditions in parallel).
#
#   bash scripts/pierdicca.sh                    # all three figures, full size (20 runs each)
#   FIGS="5" bash scripts/pierdicca.sh           # only Fig. 5 (also: FIGS="3 4")
#   RUNS=2 STEPS=20 bash scripts/pierdicca.sh    # quick look
#   JOBS=16 OUT=results/pierdicca_v2 bash scripts/pierdicca.sh
#
# Output in $OUT (default results/pierdicca), results.json / .npz / .txt each:
#   fig3/            IG vs Random vs Sweep, one UAV, Bayes only
#   fig4/<18 runs>/  IG, weights equal / biased / adaptive x r = 0..5
#   fig5/            IG / IGd x sharing x R = 5, 16 x N = 1, 2, 4, 8
# Figures: python scripts/plot.py fig3 $OUT/fig3   (fig4 $OUT/fig4, fig5 $OUT/fig5)
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

FIGS=${FIGS:-"3 4 5"}
JOBS=${JOBS:-$(nproc)}
OUT=${OUT:-results/pierdicca}
RUNS=${RUNS:-}
STEPS=${STEPS:-}
extra=()
[[ -n "$RUNS" ]] && extra+=(--runs "$RUNS")
[[ -n "$STEPS" ]] && extra+=(--steps "$STEPS")

plots=()
for fig in $FIGS; do
  case $fig in
    3)
      python scripts/run.py configs/pierdicca_fig3.yaml --planners ig random sweep \
        --jobs "$JOBS" --out "$OUT" --tag fig3 ${extra[@]+"${extra[@]}"}
      plots+=("python scripts/plot.py fig3 $OUT/fig3") ;;
    4)
      pids=()
      for w in equal biased adaptive; do for r in 0 1 2 3 4 5; do
        while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
        python scripts/run.py configs/pierdicca_fig4.yaml --planners ig --weights "$w" --r "$r" \
          --jobs 1 --out "$OUT/fig4" ${extra[@]+"${extra[@]}"} > /dev/null &
        pids+=($!)
      done; done
      for p in "${pids[@]}"; do wait "$p"; done   # any failed run stops the script (set -e)
      plots+=("python scripts/plot.py fig4 $OUT/fig4") ;;
    5)
      python scripts/run.py configs/pierdicca_fig5.yaml --planners ig igd --sharing none BS BM \
        --ranges 5 16 --n 1 2 4 8 --jobs "$JOBS" --out "$OUT" --tag fig5 ${extra[@]+"${extra[@]}"}
      plots+=("python scripts/plot.py fig5 $OUT/fig5") ;;
    *) echo "unknown figure $fig (FIGS: 3 4 5)"; exit 1 ;;
  esac
done
echo "data saved; figures (any time, no rerun):"
printf '  %s\n' "${plots[@]}"
