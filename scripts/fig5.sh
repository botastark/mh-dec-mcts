#!/usr/bin/env bash
# Paper Fig. 5: IG / IGd x sharing none / BS / BM x R = 5 / infinity x N = 2, 4, 8, plus N = 1 (38 conditions).
# One run.py call runs the whole grid; compat pierdicca: the runs of a condition in order, the
# conditions in parallel. Progress: one line per finished run.
#
#   bash scripts/fig5.sh                          # full figure: 38 conditions x 20 runs x 100 steps
#   JOBS=4 bash scripts/fig5.sh                   # 4 processes (default: nproc)
#   RUNS=2 STEPS=20 bash scripts/fig5.sh          # quick look
#
# Output: results/multi/pierdicca_fig5/ (results.json, results.txt) and paper_fig5.png
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIG=${CONFIG:-configs/pierdicca_fig5.yaml}
JOBS=${JOBS:-$(nproc)}
RUNS=${RUNS:-}
STEPS=${STEPS:-}
OUT=${OUT:-results/multi}
TAG=${TAG:-pierdicca_fig5}

extra=()
[[ -n "$RUNS" ]] && extra+=(--runs "$RUNS")
[[ -n "$STEPS" ]] && extra+=(--steps "$STEPS")

python scripts/run.py "$CONFIG" --planners ig igd --sharing none BS BM --ranges 5 16 --n 1 2 4 8 \
  --jobs "$JOBS" --out "$OUT" --tag "$TAG" ${extra[@]+"${extra[@]}"}
python scripts/plot_fig5.py "$OUT/$TAG"
