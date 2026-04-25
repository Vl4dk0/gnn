#!/bin/bash
#SBATCH --job-name=cage-vol-h64
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

uv run python -m ai.cage.voltage.rl_train --steps 99999999 --hidden-dim 64 --name gin_vol_h64_ppo
