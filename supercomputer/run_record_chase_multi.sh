#!/bin/bash
#SBATCH --job-name=record_chase_multi
#SBATCH --output=benchmark_runs/%x_%j.out
#SBATCH --error=benchmark_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=512G
#SBATCH --time=25:00:00

# Run ALL record-chase targets concurrently inside a SINGLE node/job.
#
# Why one job instead of five: the perun2501173 association caps each user at 4
# concurrent NODES (node=4), and the standing GPU training jobs already hold 3.
# Five separate whole-node CPU jobs would queue forever on AssocGrpNodeLimit, and
# five small jobs would each be scheduled onto a distinct idle node, re-hitting
# the same cap. Packing all chases into one node-job uses exactly 1 CPU node
# (1 + 3 GPU = 4, within the limit) and runs every target in parallel.
#
# 256 cores split as 5 targets x 48 workers = 240 (+ 5 coordinators + OS slack).
#
# Output: results/records/<stamp>_kK_gG_VARIANT/ per target (graph6 per girth,
# records.json, summary.md). Per-target stdout -> benchmark_runs/rc_*_<jobid>.log.
# results/ is gitignored: retrieve with `git add -f results/records/<dir>`,
# commit + push from PERUN, git pull locally, then merge with
#   uv run python -m ai.cage.record_chase merge results/records/<d1> ... --out results/records/merged

set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/gnn

WORKERS=48
BUDGET=85500  # 23h45m, leaves ~75min under the 25h wall to write summaries

# Caps raised hard for a record search: the producer keeps feeding (1000 lifts),
# runs never hit a step cap (governed by the time budget), and refine/excision
# work much harder per candidate. The group-order cap in the voltage env was
# removed entirely (see ai/cage/voltage/rl/env.py).
MAX_CANDIDATES=1000
MAX_TOTAL_STEPS=100000000
REFINE_MAX_ITER=10000
EXCISION_BACKTRACKS=5000

chase() {  # args: k g variant
    uv run python -u -m ai.cage.record_chase chase "$1" "$2" \
        --variant "$3" \
        --workers "$WORKERS" \
        --time-budget "$BUDGET" \
        --max-candidates "$MAX_CANDIDATES" \
        --max-total-steps "$MAX_TOTAL_STEPS" \
        --refine-max-iter "$REFINE_MAX_ITER" \
        --excision-backtracks "$EXCISION_BACKTRACKS" \
        --out results/records \
        > "benchmark_runs/rc_$1_$2_$3_${SLURM_JOB_ID}.log" 2>&1 &
}

chase 8 5 full
chase 8 5 no_refine
chase 9 5 full
chase 10 5 full
chase 11 6 full

wait
echo "==> all record chases done (job $SLURM_JOB_ID)"
