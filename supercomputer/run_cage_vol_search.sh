#!/bin/bash
#SBATCH --job-name=cage-vol-search
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

# Run meta search (random + tabu) for various targets — each independent
uv run python -m ai.cage.voltage.search 3 5
uv run python -m ai.cage.voltage.search 3 6
uv run python -m ai.cage.voltage.search 3 7
uv run python -m ai.cage.voltage.search 3 8
uv run python -m ai.cage.voltage.search 3 9
uv run python -m ai.cage.voltage.search 3 10
uv run python -m ai.cage.voltage.search 3 11
uv run python -m ai.cage.voltage.search 3 12
