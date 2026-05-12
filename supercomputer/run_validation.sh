#!/bin/bash
#SBATCH --job-name=validation
#SBATCH --output=training_runs/%x_%j.out
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Full side-by-side validation of every trained model.
# Starts the Flask backend on localhost:5555, runs validation.py
# against it, then shuts the backend down. Outputs go to
# training_runs/validation/<timestamp>/ (not gitignored), so the
# results can be committed straight from PERUN.

set -euo pipefail

source .activate_scratch
cd ~/gnn

PORT="${PORT:-5555}"
BACKEND_LOG="training_runs/validation_backend_${SLURM_JOB_ID:-local}.log"

echo "==> launching backend on 127.0.0.1:${PORT}"
DEBUG=False HOST=127.0.0.1 PORT="${PORT}" uv run python -m backend.app \
    > "${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
trap 'kill ${BACKEND_PID} 2>/dev/null || true; wait ${BACKEND_PID} 2>/dev/null || true' EXIT

echo "==> waiting for backend (pid=${BACKEND_PID})"
for i in $(seq 1 120); do
    if curl -sf -m 2 "http://127.0.0.1:${PORT}/api/config" > /dev/null 2>&1; then
        echo "==> backend ready after ${i}s"
        break
    fi
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "ERROR: backend died during startup; last log lines:"
        tail -40 "${BACKEND_LOG}"
        exit 1
    fi
    sleep 1
done

if ! curl -sf -m 2 "http://127.0.0.1:${PORT}/api/config" > /dev/null 2>&1; then
    echo "ERROR: backend never responded within 120s; last log lines:"
    tail -40 "${BACKEND_LOG}"
    exit 1
fi

echo "==> running validation"
uv run python -u validation.py \
    --task all \
    --cage-budget 60 \
    --cage-seeds 2 \
    --output-root training_runs/validation \
    --api "http://127.0.0.1:${PORT}"

echo "==> validation done; backend will be terminated by EXIT trap"
