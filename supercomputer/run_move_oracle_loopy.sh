#!/bin/bash
#SBATCH --job-name=move-oracle-loopy
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd ~/gnn

# Loopy-backbone variant of the swap scorer for post-lift tabu refinement.
# Same task as run_move_oracle.sh (predict Δ short-cycle-cost for 2-switches),
# but the encoder is the cycle-aware Loopy_GNN instead of GINEConv -- the
# hypothesis being that a cycle-sensitive backbone helps a cycle-cost task.
# --backbone loopy attaches r-neighborhood cycle tensors per graph.
# Saves to ai/trained/move_oracle_loopy_r3/ (does NOT touch the gine oracle).
#
# Sizing: the loopy path is ~6x slower per epoch (one tiny graph per forward,
# no PyG batching), so 200k samples x 50 epochs would blow past any wall.
# 50k samples + 30 dataset-gen workers (32 cpus) keeps dataset gen to a few
# minutes and 50 epochs well inside 12h; training checkpoints the best model to
# disk on every val improvement, so a wall-kill still leaves a usable model.

uv run python -u -m ai.cage.refine.train \
  --backbone loopy --r 3 \
  --samples 50000 --epochs 50 --batch-size 64 \
  --hidden-dim 128 --num-layers 4 --lr 1e-3 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --workers 30 \
  --seed 42
