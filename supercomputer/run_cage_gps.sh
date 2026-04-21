#!/bin/bash
#SBATCH --job-name=gnn-cage-gps
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source .activate_scratch

cd ~/gnn

uv run python -m ai.cage.rl.train --model gps --conv-type gin --hidden-dim 64 --num-layers 4 --steps 300000 --heads 4
