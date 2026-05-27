#!/bin/bash
#SBATCH --job-name=excision-repair
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source .activate_scratch

cd ~/gnn

# PPO repair policy for tree excision (depth 1, Petersen + Heawood bank).
# Compares against the classical backtracking baseline on the same instances.
# Saves to ai/trained/excision_repair/excision_repair_policy/.

uv run python -u -m ai.cage.excision.train \
  --episodes 5000 --g-target 4 --depth 1 \
  --hidden-dim 128 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --seed 42
