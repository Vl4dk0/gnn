#!/bin/bash
#SBATCH --job-name=voltage-predictor
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source .activate_scratch

cd ~/gnn

# Train girth predictors for several g_targets.
# Each gets 200k samples, 100 epochs, hidden_dim=128.
# Dataset generation is CPU-bound (deduplicated); training is GPU.

uv run python -u -m ai.cage.voltage.train --k 3 --g 5  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --k 3 --g 6  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --k 3 --g 7  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --k 3 --g 8  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 80  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --k 3 --g 9  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 100 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --k 3 --g 10 --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 100 --print-every 5 --seed 42
