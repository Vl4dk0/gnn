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

# PPO repair policy for tree excision.
# Trains on guaranteed-solvable matching-removal instances. The easy rung
# (synthetic cages, g=5, match-size 2) saturated at ~100% success, so this is
# the harder rung: lift-sized source graphs, stricter g=6 legality, and a
# larger match (more simultaneous deficiencies) — solvable but non-trivial
# (~60-70%+ early and climbing). Saves to ai/trained/excision_repair/.

uv run python -u -m ai.cage.excision.train \
  --episodes 30000 --g-target 6 --depth 1 \
  --instance-source lifts --match-size 4 \
  --hidden-dim 128 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --seed 42
