#!/bin/bash
#SBATCH --job-name=cage-vol-search
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

uv run python -m ai.cage.voltage.search 3 5 --workers 30
uv run python -m ai.cage.voltage.search 3 6 --workers 30
uv run python -m ai.cage.voltage.search 3 7 --workers 30
uv run python -m ai.cage.voltage.search 3 8 --workers 30
uv run python -m ai.cage.voltage.search 3 9 --workers 30 --max-group-order 200
uv run python -m ai.cage.voltage.search 3 10 --workers 30 --max-group-order 200
uv run python -m ai.cage.voltage.search 3 11 --workers 30 --max-group-order 300
uv run python -m ai.cage.voltage.search 3 12 --workers 30 --max-group-order 300
