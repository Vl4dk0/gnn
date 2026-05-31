#!/bin/bash
#SBATCH --job-name=rc_algebraic
#SBATCH --output=benchmark_runs/%x_%j.out
#SBATCH --error=benchmark_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=512G
#SBATCH --time=25:00:00

# EXPERIMENT: do the stuck k!=3 targets fare better with the CLASSICAL
# voltage_algebraic producer than the default voltage_rl?
#
# Rationale: the voltage_rl policy has action dim 100, so for the group orders
# these high-k targets require (e.g. ~111 for (11,6)) it is out-of-distribution
# and degrades to near-random. After 3h the voltage_rl baseline (job 48420)
# found ZERO girth-5+ graphs for (8,5)/(9,5)/(10,5)/(11,6). voltage_algebraic is
# classical tabu over voltages with no action-dim limit. This run goes head to
# head with the baseline on the same targets; compare girth-5+ yield.

set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/gnn

WORKERS=60
BUDGET=85500
PRODUCER=voltage_algebraic
MAX_CANDIDATES=1000
MAX_TOTAL_STEPS=100000000
REFINE_MAX_ITER=10000
EXCISION_BACKTRACKS=5000

chase() {  # args: k g
    uv run python -u -m ai.cage.record_chase chase "$1" "$2" \
        --variant full --producer "$PRODUCER" \
        --workers "$WORKERS" --time-budget "$BUDGET" \
        --max-candidates "$MAX_CANDIDATES" --max-total-steps "$MAX_TOTAL_STEPS" \
        --refine-max-iter "$REFINE_MAX_ITER" --excision-backtracks "$EXCISION_BACKTRACKS" \
        --out results/records \
        > "benchmark_runs/rc_$1_$2_${PRODUCER}_${SLURM_JOB_ID}.log" 2>&1 &
}

chase 8 5
chase 9 5
chase 10 5
chase 11 6

wait
echo "==> algebraic-producer experiment done (job $SLURM_JOB_ID)"
