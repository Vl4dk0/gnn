#!/bin/bash
#SBATCH --job-name=group-promise
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd ~/gnn

# Group-promise predictor — the genuine ML heuristic for Cayley cage search.
# It predicts the best girth achievable by ANY degree-k generating set of a
# group, so the meta-search can prune the catalogue before paying for the
# (expensive) classical inner search.
#
# Dataset generation runs the classical inner search per group and is
# CPU-heavy: --gen-workers spreads it over the allocated cores. Model
# training then runs on the GPU.

uv run python -u -m ai.cage.cayley.group_train \
  --targets "3,5;3,6;3,7;3,8;4,6;4,7;5,5;5,6" \
  --max-group-order 1400 \
  --gen-random-trials 1000 \
  --gen-tabu-iters 600 \
  --gen-workers 30 \
  --epochs 200 \
  --batch-size 32 \
  --hidden-dim 96 \
  --num-layers 4 \
  --weight-decay 1e-4 \
  --print-every 5 \
  --seed 42
