#!/bin/bash
#SBATCH --job-name=benchmark_s100
#SBATCH --output=benchmark_runs/%x_%j.out
#SBATCH --error=benchmark_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=256
#SBATCH --mem=512G
#SBATCH --time=48:00:00

# Overnight, high-seed-count variant of run_benchmark.sh: the full in-process,
# multicore benchmark of every trained model / construction approach:
#   - degree, min_cycle   (all GNN architectures, on a shared graph battery)
#   - cage generators      (randomwalk, astar, bruteforce, rl, voltage_rl, and
#                           voltage with NO-GNN vs GNN girth-predictor guidance,
#                           plus forge ablations)
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
#
# Timing: the seeds=5 full run is ~17-19 min (956 trials). Trial count scales
# linearly with seeds, so seeds=100 is ~20x => ~6h wall at 254 workers. --time is
# set to the 48h maximum purely as headroom so the job is never killed early.

set -uo pipefail

cd ~/gnn

# --seeds 100: each cage target is run 100 times so the (k,g) matrix reports a
# statistically solid mean time-to-solve / size (the stochastic generators vary
# run to run). This is the high-confidence overnight configuration.
uv run python -u -m results.runner \
    --benchmarks degree,min_cycle,cage,refine,excision \
    --seeds 100 \
    --cage-budget 60 \
    --cage-max-steps 200000 \
    --task-timeout 120 \
    --workers 254 \
    --out-root results/runs

echo "==> benchmark done"
