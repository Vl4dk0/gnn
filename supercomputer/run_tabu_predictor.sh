#!/bin/bash
#SBATCH --job-name=tabu-predictor
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

# Tabu-cost predictor: same architecture and features as the girth predictor,
# but regresses the dense short-walk cost the tabu search minimizes (lower is
# better) instead of the lift's girth. Ranked ASCENDING in beam search.
# Saves to ai/trained/voltage_girth/tabu_predictor/ alongside girth_predictor so
# the search, registry, and backend can A/B the two.

uv run python -u -m ai.cage.voltage.train \
  --targets "3,5;3,6;3,7;3,8;3,9;3,10;4,5;4,6;4,7;4,8;5,5;5,6;5,7" \
  --kind tabu_cost \
  --samples 1000000 --epochs 100 --batch-size 256 \
  --hidden-dim 192 --num-layers 6 --max-group-order 100 \
  --lr 1e-3 --weight-decay 1e-4 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --workers 8 \
  --model-id tabu_predictor \
  --print-every 5 --seed 42
