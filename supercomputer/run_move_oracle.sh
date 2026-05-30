#!/bin/bash
#SBATCH --job-name=move-oracle
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

source .activate_scratch

cd ~/gnn

# Supervised swap scorer for post-lift tabu refinement.
# Predicts Δ(short-cycle-cost) for candidate 2-switches; the TabuRefiner uses
# this to rank swaps instead of brute-forcing each delta classically.
# Saves to ai/trained/move_oracle/move_oracle/.

uv run python -u -m ai.cage.refine.train \
  --samples 200000 --epochs 50 --batch-size 64 \
  --hidden-dim 128 --num-layers 4 --lr 1e-3 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --workers 8 \
  --seed 42
