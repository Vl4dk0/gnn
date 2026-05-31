#!/bin/bash
#SBATCH --job-name=bench_forge
#SBATCH --output=benchmark_runs/%x_%j.out
#SBATCH --error=benchmark_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=256
#SBATCH --mem=512G
#SBATCH --time=02:00:00

# Forge-scoped benchmark, aimed at the defect-density refinement gate.
#
# Runs ONLY the forge cage approach (all its variants) across the full (k,g)
# target grid. Among the forge variants are:
#   - voltage_rl              -> the girth-margin gate baseline (default producer)
#   - defect_gate_tau=0.1/0.2/0.3 -> the new size-relative defect-density gate
#                                    (same default producer + full pipeline)
# so the gate is the only thing that varies between baseline and the sweep,
# giving a clean head-to-head of girth-margin vs defect-density routing.
#
# Workers pinned at 128 (the value proven stable on this node; higher counts
# starved memory bandwidth and slowed time-budgeted trials into failure).
# Outputs in results/runs/<timestamp>/ (gitignored -> force-add to retrieve).

set -uo pipefail

cd ~/gnn

uv run python -u -m results.runner \
    --benchmarks cage \
    --approaches forge \
    --seeds 40 \
    --cage-budget 30 \
    --cage-max-steps 200000 \
    --task-timeout 90 \
    --workers 128 \
    --out-root results/runs

echo "==> forge benchmark done"
