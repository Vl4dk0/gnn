#!/bin/bash
#SBATCH --job-name=cage-vol-search-ml
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source .activate_scratch

cd ~/gnn

# ML-guided meta search: random + tabu (parallel) + beam search (with predictor)
# Each (k, g) is searched once per predictor variant for thesis comparison:
#   A: girth_predictor_k{k}_g{g}      (per-(k,g) — narrowest specialist)
#   B: girth_predictor_g{g}_multik    (per-g — generalizes over k)
#   C: girth_predictor_unified        (single model across full grid)
# Requires that run_voltage_predictor.sh, run_voltage_predictor_per_g.sh, and
# run_voltage_predictor_unified.sh have all completed.

# --- k=3 ---
uv run python -m ai.cage.voltage.search 3 5  --workers 30 --model-id girth_predictor_k3_g5
uv run python -m ai.cage.voltage.search 3 5  --workers 30 --model-id girth_predictor_g5_multik
uv run python -m ai.cage.voltage.search 3 5  --workers 30 --model-id girth_predictor_unified

uv run python -m ai.cage.voltage.search 3 6  --workers 30 --model-id girth_predictor_k3_g6
uv run python -m ai.cage.voltage.search 3 6  --workers 30 --model-id girth_predictor_g6_multik
uv run python -m ai.cage.voltage.search 3 6  --workers 30 --model-id girth_predictor_unified

uv run python -m ai.cage.voltage.search 3 7  --workers 30 --model-id girth_predictor_k3_g7
uv run python -m ai.cage.voltage.search 3 7  --workers 30 --model-id girth_predictor_g7_multik
uv run python -m ai.cage.voltage.search 3 7  --workers 30 --model-id girth_predictor_unified

uv run python -m ai.cage.voltage.search 3 8  --workers 30 --model-id girth_predictor_k3_g8
uv run python -m ai.cage.voltage.search 3 8  --workers 30 --model-id girth_predictor_g8_multik
uv run python -m ai.cage.voltage.search 3 8  --workers 30 --model-id girth_predictor_unified

uv run python -m ai.cage.voltage.search 3 9  --workers 30 --max-group-order 200 --model-id girth_predictor_k3_g9
uv run python -m ai.cage.voltage.search 3 9  --workers 30 --max-group-order 200 --model-id girth_predictor_g9_multik
uv run python -m ai.cage.voltage.search 3 9  --workers 30 --max-group-order 200 --model-id girth_predictor_unified

uv run python -m ai.cage.voltage.search 3 10 --workers 30 --max-group-order 200 --model-id girth_predictor_k3_g10
uv run python -m ai.cage.voltage.search 3 10 --workers 30 --max-group-order 200 --model-id girth_predictor_g10_multik
uv run python -m ai.cage.voltage.search 3 10 --workers 30 --max-group-order 200 --model-id girth_predictor_unified
