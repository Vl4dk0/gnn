#!/bin/bash
#SBATCH --job-name=cage-vol-gps128-8h
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

uv run python -u -m ai.cage.voltage.rl_train --steps 99999999 --hidden-dim 128 --conv-type gps --heads 8 --name gps_vol_h128_8h_ppo --save-every 100000
