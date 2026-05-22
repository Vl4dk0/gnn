#!/bin/bash
#SBATCH --job-name=cayley-baseline
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd ~/gnn

# Classical (no-ML) Cayley graph cage finder — the baseline every
# ML-assisted result must beat. Pure random + tabu search over a catalogue
# of finite groups, parallelised across CPU cores. No GPU, no neural net.
#
# Results table (smallest Cayley (k,g)-graph found vs. known cage order) is
# printed to stdout and written as JSON for later comparison.

uv run python -u -m ai.cage.cayley.baseline \
  --targets "3,5;3,6;3,7;3,8;3,9;3,10;3,11;3,12;4,6;4,7;4,8;5,5;5,6" \
  --max-order 2200 \
  --trials 4000 \
  --tabu-iters 3000 \
  --workers 30 \
  --seed 42 \
  --json training_runs/cayley_baseline_results.json
