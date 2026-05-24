#!/bin/bash
#SBATCH --job-name=cayley-comparison
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd ~/gnn

# Classical baseline vs group-guided Cayley search.
#
# Reuses the cached classical results from run_cayley_baseline.sh
# (training_runs/cayley_baseline_results.json), so we don't repay the
# ~11h baseline wall-clock. For each of the 8 trained targets the
# harness reports:
#   * the predictor's predicted girth for the baseline-winning group
#     and its rank in the catalogue (predictor quality), and
#   * the guided search at multiple prune_slack values, with order
#     found, wall-clock, groups kept, and whether the same order as
#     the classical baseline was reached.
#
# Slack sweep makes the "how aggressively can we prune before losing
# the optimum?" curve explicit — that's the interesting result.

uv run python -u -m ai.cage.cayley.group_search compare \
  --targets "3,5;3,6;3,7;3,8;4,6;4,7;5,5;5,6" \
  --model-id group_promise_predictor_multi \
  --baseline-json training_runs/cayley_baseline_results.json \
  --slacks "0.0,0.5,1.0,2.0" \
  --max-order 2200 \
  --trials 4000 \
  --tabu-iters 3000 \
  --workers 30 \
  --seed 42 \
  --output-json training_runs/cayley_comparison_results.json
