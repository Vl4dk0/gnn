#!/bin/bash
#SBATCH --job-name=cage-vol-search-ml
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source .activate_scratch

cd ~/gnn

# ML-guided meta search: random + tabu (parallel) + beam search (with predictor)
# Uses the single (k, g)-independent girth predictor for every target.
# Requires that run_voltage_predictor.sh has completed.

# --- k=3 ---
uv run python -m ai.cage.voltage.search 3 5  --workers 30 --model-id girth_predictor
uv run python -m ai.cage.voltage.search 3 6  --workers 30 --model-id girth_predictor
uv run python -m ai.cage.voltage.search 3 7  --workers 30 --model-id girth_predictor
uv run python -m ai.cage.voltage.search 3 8  --workers 30 --model-id girth_predictor
uv run python -m ai.cage.voltage.search 3 9  --workers 30 --max-group-order 200 --model-id girth_predictor
uv run python -m ai.cage.voltage.search 3 10 --workers 30 --max-group-order 200 --model-id girth_predictor
