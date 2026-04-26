#!/bin/bash
#SBATCH --job-name=voltage-predictor-unified
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source .activate_scratch

cd ~/gnn

# Variant C: single unified girth predictor across the full (k, g) grid.
# Tests whether one model can subsume all targets via context features.
# Larger sample budget and slightly larger hidden dim absorb the extra variance.

uv run python -u -m ai.cage.voltage.train \
  --targets "3,5;3,6;3,7;3,8;3,9;3,10;4,5;4,6;4,7;4,8;5,5;5,6;5,7" \
  --samples 600000 --epochs 100 --batch-size 256 \
  --hidden-dim 192 --num-layers 4 --max-group-order 100 \
  --print-every 5 --seed 42
