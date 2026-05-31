#!/bin/bash
#SBATCH --job-name=excision
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source .activate_scratch

cd ~/gnn

# PPO repair policy for tree excision.
# Trains on guaranteed-solvable matching-removal instances. Curriculum so far:
# g=5/match-2 and g=6/match-4 both saturated at ~100% success, so this is the
# next rung: g=7 legality (very strict) with match-size 6 (many simultaneous
# deficiencies). Smoke shows ~30-35% early and climbing — hard but learnable,
# the informative regime. (lifts can't reach g=7, so synthetic on the McGee +
# Tutte-Coxeter cages.) Saves to ai/trained/excision/.

uv run python -u -m ai.cage.excision.train \
  --episodes 300000 --g-target 7 --depth 1 \
  --instance-source synthetic --match-size 6 \
  --hidden-dim 128 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --seed 42
