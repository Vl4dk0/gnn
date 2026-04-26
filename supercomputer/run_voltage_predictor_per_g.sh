#!/bin/bash
#SBATCH --job-name=voltage-predictor-per-g
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

# Variant B: per-g predictors. One model per girth target, trained over k=3,4,5.
# Tests whether the predictor generalizes across degrees.

uv run python -u -m ai.cage.voltage.train --targets "3,5;4,5;5,5"  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,6;4,6;5,6"  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,7;4,7;5,7"  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,8;4,8;5,8"  --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 80  --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,9;4,9"      --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 100 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,10;4,10"    --samples 200000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 100 --print-every 5 --seed 42
