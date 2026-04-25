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

# Train supervised girth predictor for multiple targets
echo "=== Training girth predictors ==="
uv run python -m ai.cage.voltage.train --k 3 --g 7 --samples 20000 --epochs 100 --hidden-dim 128 &&
uv run python -m ai.cage.voltage.train --k 3 --g 8 --samples 20000 --epochs 100 --hidden-dim 128 &&
uv run python -m ai.cage.voltage.train --k 3 --g 9 --samples 20000 --epochs 100 --hidden-dim 128 &&
uv run python -m ai.cage.voltage.train --k 3 --g 10 --samples 20000 --epochs 100 --hidden-dim 128

# Run meta search (random + tabu) for various targets
echo "=== Running meta search ==="
uv run python -m ai.cage.voltage.search 3 5 &&
uv run python -m ai.cage.voltage.search 3 6 &&
uv run python -m ai.cage.voltage.search 3 7 &&
uv run python -m ai.cage.voltage.search 3 8 &&
uv run python -m ai.cage.voltage.search 3 9 &&
uv run python -m ai.cage.voltage.search 3 10
