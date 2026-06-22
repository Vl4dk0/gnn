#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark_runs/%x_%j.out
#SBATCH --error=benchmark_runs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=256
#SBATCH --mem=512G
#SBATCH --time=08:00:00

# In-process, multicore benchmark of every trained model / construction approach:
#   - degree, min_cycle   (all GNN architectures, on a shared graph battery)
#   - cage generators      (randomwalk, astar, bruteforce, rl, voltage_rl, and
#                           voltage with NO-GNN vs GNN girth-predictor guidance)
#   - refine               (tabu: NO-GNN exact vs GNN MoveOracle)
#   - excision             (repair: classical vs RL policy)
#
# Runs models directly in-process (NO Flask backend) and fans out every trial
# across CPU cores via a ProcessPoolExecutor. Workers are pinned to CPU + 1
# torch thread each, so this belongs on the CPU partition, not the GPU one.
#
# Outputs land in results/runs/<timestamp>/ (raw.json, results.jsonl,
# summary.md, per-benchmark CSVs). results/runs/ is gitignored, so to retrieve
# a run, force-add that run dir from PERUN (git add -f results/runs/<stamp>),
# commit, push, then git pull locally.

set -uo pipefail

cd ~/gnn

# Per-trial wall-clock budget for cage generators (seconds), overridable per
# submission via --export so the same script can run several budget sweeps.
# TASK_TIMEOUT is the hard SIGALRM backstop and must exceed CAGE_BUDGET.
CAGE_BUDGET="${CAGE_BUDGET:-300}"
TASK_TIMEOUT="${TASK_TIMEOUT:-360}"

# --seeds 5: each cage target is run 5 times so the (k,g) matrix can report a
# meaningful mean time-to-solve (the stochastic generators vary run to run).
uv run python -u -m results.runner \
    --benchmarks degree,min_cycle,cage,refine,excision \
    --seeds 5 \
    --cage-budget "${CAGE_BUDGET}" \
    --cage-max-steps 200000 \
    --task-timeout "${TASK_TIMEOUT}" \
    --workers 128 \
    --out-root results/runs

echo "==> benchmark done"
