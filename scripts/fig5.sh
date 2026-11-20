#!/usr/bin/env bash
# Paper Fig. 5: IG / IGd x sharing none / BS / BM x R = 5 / infinity x N = 2, 4, 8, plus N = 1 (38 conditions).
# Progress: one line per finished run; the full figure takes ~50x as long as RUNS=2 STEPS=20.
#
#   bash scripts/fig5.sh                          # full figure: 37 conditions x 20 runs x 100 steps
#   JOBS=4 bash scripts/fig5.sh                   # run 4 conditions in parallel (default: nproc)
#   RUNS=2 STEPS=20 bash scripts/fig5.sh          # quick look
#   CHECK=1 PA=~/repos/Precision-Agriculture-Dev bash scripts/fig5.sh
#       first: short runs of Pierdicca's own code for N = 2 and 4 (all planners / sharing / ranges),
#       compared step by step with the new code; stops if any condition is DIFFERENT.
#
# Output: results/multi/pierdicca_fig5/*.json and paper_fig5.png
set -euo pipefail
# one maths thread per process: the conditions already run in parallel
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIG=${CONFIG:-configs/pierdicca_fig5.yaml}
JOBS=${JOBS:-$(nproc)}
RUNS=${RUNS:-}
STEPS=${STEPS:-}
OUT=${OUT:-results/multi}
CHECK=${CHECK:-0}
PA=${PA:-}
CHECK_RUNS=${CHECK_RUNS:-3}
CHECK_STEPS=${CHECK_STEPS:-30}

extra=()
[[ -n "$RUNS" ]] && extra+=(--runs "$RUNS")
[[ -n "$STEPS" ]] && extra+=(--steps "$STEPS")

# ------------------------------------------------------------------ 0. check against his code
if [[ "$CHECK" == 1 ]]; then
  [[ -z "$PA" ]] && { echo "set PA=<path to Precision-Agriculture-Dev>"; exit 1; }
  mkdir -p "$OUT/pa_check"
  for p in ig igd; do for s in none BS BM; do for R in 5 16; do for n in 2 4; do
    f="$OUT/pa_check/${p}_${s}_R${R}_N${n}.npz"
    [[ -f "$f" ]] || python scripts/run_pa_multi.py "$PA" $p $s $R $n "$CHECK_RUNS" "$CHECK_STEPS" "$f" > /dev/null
    line=$(python scripts/check_vs_pierdicca_multi.py "$CONFIG" $p $s $R $n "$f")
    echo "$line"
    [[ "$line" == *IDENTICAL ]] || { echo "STOP: differs from Pierdicca's code"; exit 1; }
  done; done; done; done
  echo "all checked conditions IDENTICAL"
fi

# ------------------------------------------------------------------ 1. new code, all conditions
conditions=()
for p in ig igd; do
  conditions+=("$p none 5 1")                          # N = 1: sharing and range do not matter
  for s in none BS BM; do for R in 5 16; do for n in 2 4 8; do
    conditions+=("$p $s $R $n")
  done; done; done
done
run_one() {  # planner sharing range n
  python scripts/run_multi.py "$CONFIG" --planners "$1" --sharing "$2" --ranges "$3" --n "$4" \
    --out "$OUT" ${extra[@]+"${extra[@]}"}
}
echo "running ${#conditions[@]} conditions, $JOBS at a time"
for c in "${conditions[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done   # a failed condition stops the script
  run_one $c &
done
wait

# ------------------------------------------------------------------ 2. figure
name=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CONFIG")
python scripts/plot_paper_fig5.py "$OUT/$name"
