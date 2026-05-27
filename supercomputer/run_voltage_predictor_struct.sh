#!/bin/bash
#SBATCH --job-name=voltage-predictor-struct
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source .activate_scratch

cd ~/gnn

# Girth predictor with structural node features (cycle counts + RWPE).
# Mirrors run_voltage_predictor.sh exactly except for the structural-feature
# flags. Set --model-id manually when submitting so this run does not clobber
# the baseline at ai/trained/voltage_girth/girth_predictor/.
#
# Example submit:
#   sbatch --export=ALL,MODEL_ID=girth_predictor_struct \
#          supercomputer/run_voltage_predictor_struct.sh

uv run python -u -m ai.cage.voltage.train \
  --targets "3,5;3,6;3,7;3,8;3,9;3,10;4,5;4,6;4,7;4,8;5,5;5,6;5,7" \
  --samples 1000000 --epochs 100 --batch-size 256 \
  --hidden-dim 192 --num-layers 4 --max-group-order 100 \
  --weight-decay 1e-4 \
  --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8 \
  --workers 8 \
  --model-id "${MODEL_ID:?set MODEL_ID, e.g. MODEL_ID=girth_predictor_struct}" \
  --print-every 5 --seed 42
