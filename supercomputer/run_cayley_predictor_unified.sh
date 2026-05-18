#!/bin/bash
#SBATCH --job-name=cayley-predictor-unified
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd ~/gnn

# Unified Cayley girth predictor across multiple (k, g) targets.
# One model handles every target in the list — no per-(k,g) retraining.
# Each sample carries its own (k, g_target) as graph-level context.

uv run python -u -m ai.cage.cayley.train \
  --targets "3,6;3,7;3,8;3,9;3,10;4,6;4,7;4,8" \
  --samples 200000 --epochs 100 --batch-size 256 \
  --hidden-dim 192 --num-layers 4 --max-group-order 200 \
  --weight-decay 1e-4 \
  --print-every 5 --seed 42
