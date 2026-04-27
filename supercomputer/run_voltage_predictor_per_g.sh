#!/bin/bash
#SBATCH --job-name=voltage-predictor-per-g-v2
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

# Variant B retrain (v2): per-g predictors with bigger samples + weight decay
# to combat the overfitting observed in v1 (train loss collapsed to 0.013 while
# val loss climbed to 0.20-0.40 by epoch 100).

uv run python -u -m ai.cage.voltage.train --targets "3,5;4,5;5,5"  --samples 300000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --weight-decay 1e-4 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,6;4,6;5,6"  --samples 300000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --weight-decay 1e-4 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,7;4,7;5,7"  --samples 300000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 60  --weight-decay 1e-4 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,8;4,8;5,8"  --samples 300000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 80  --weight-decay 1e-4 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,9;4,9"      --samples 300000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 100 --weight-decay 1e-4 --print-every 5 --seed 42
uv run python -u -m ai.cage.voltage.train --targets "3,10;4,10"    --samples 300000 --epochs 100 --batch-size 128 --hidden-dim 128 --num-layers 4 --max-group-order 100 --weight-decay 1e-4 --print-every 5 --seed 42
