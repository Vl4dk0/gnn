#!/bin/bash
#SBATCH --job-name=gnn-cage
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source .activate_scratch

cd ~/gnn

uv run python -m ai.cage.rl.train --model gin --hidden-dim 128 --num-layers 6 --steps 500000 --name h128_l6_ppo
