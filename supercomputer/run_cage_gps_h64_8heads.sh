#!/bin/bash
#SBATCH --job-name=cage-gps-h64-8h
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

uv run python -m ai.cage.rl.train --model gps --conv-type gin --hidden-dim 64 --num-layers 4 --steps 99999999 --heads 8 --name gps_gin_8h_ppo
