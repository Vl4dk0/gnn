#!/bin/bash
#SBATCH --job-name=move-oracle-loopy
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

cd ~/gnn

# Loopy-backbone variant of the swap scorer for post-lift tabu refinement.
# Same task as run_move_oracle.sh (predict Δ short-cycle-cost for 2-switches),
# but the encoder is the cycle-aware Loopy_GNN instead of GINEConv -- the
# hypothesis being that a cycle-sensitive backbone helps a cycle-cost task.
# --backbone loopy attaches r-neighborhood cycle tensors per graph.
# Saves to ai/trained/move_oracle_loopy_r3/ (does NOT touch the gine oracle).

uv run python -u -m ai.cage.refine.train \
  --backbone loopy --r 3 \
  --samples 200000 --epochs 50 --batch-size 64 \
  --hidden-dim 128 --num-layers 4 --lr 1e-3 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --workers 8 \
  --seed 42
