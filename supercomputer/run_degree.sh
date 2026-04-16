#!/bin/bash
#SBATCH --job-name=gnn-degree
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

uv run python -m ai.degree.train --model gcn --hidden-dim 128 --num-layers 6 --epochs 5000 --name gcn_h128_l6 &&
	uv run python -m ai.degree.train --model sage --hidden-dim 128 --num-layers 6 --epochs 5000 --name sage_h128_l6 &&
	uv run python -m ai.degree.train --model gin --hidden-dim 128 --num-layers 6 --epochs 5000 --name gin_h128_l6 &&
	uv run python -m ai.degree.train --model gps --conv-type gcn --hidden-dim 128 --num-layers 6 --heads 8 --epochs 5000 --name gps_gcn_h128_l6 &&
	uv run python -m ai.degree.train --model gps --conv-type gin --hidden-dim 128 --num-layers 6 --heads 8 --epochs 5000 --name gps_gin_h128_l6 &&
	uv run python -m ai.degree.train --model gps --conv-type sage --hidden-dim 128 --num-layers 6 --heads 8 --epochs 5000 --name gps_sage_h128_l6 &&
	uv run python -m ai.degree.train --model loopy --hidden-dim 128 --num-layers 6 --r 3 --epochs 5000 --name loopy_h128_l6
