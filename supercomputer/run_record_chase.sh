#!/bin/bash
#SBATCH --job-name=record_chase
#SBATCH --output=benchmark_runs/%x_%j.out
#SBATCH --error=benchmark_runs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501173
#SBATCH --qos=perun2501173
#SBATCH --cpus-per-task=256
#SBATCH --mem=512G
#SBATCH --time=25:00:00

# Record-chasing harness: drive forge in a multiprocess restart loop to hunt for
# the smallest (k,g')-graphs it can build, persisting every improvement.
#
# Usage:  sbatch --job-name=rc_K_G_VARIANT run_record_chase.sh K G [VARIANT]
#   e.g.  sbatch --job-name=rc_8_5_full     run_record_chase.sh 8 5 full
#         sbatch --job-name=rc_8_5_norefine run_record_chase.sh 8 5 no_refine
#
# VARIANT is "full" (default) or "no_refine" (disables the refine stage).
#
# forge is CPU-bound and its GNNs are CPU-pinned, so this belongs on the CPU
# partition (256 cores) not the GPU one. Each worker runs single-threaded.
#
# Output lands in results/records/<stamp>_kK_gG_VARIANT/ (graph6 per girth,
# records.json, summary.md). results/ is gitignored, so to retrieve a run:
# git add -f results/records/<dir> from PERUN, commit, push, then git pull
# locally, then merge with:
#   uv run python -m ai.cage.record_chase merge results/records/<dir1> ... --out results/records/merged

set -uo pipefail

K=${1:?usage: run_record_chase.sh K G [VARIANT]}
G=${2:?usage: run_record_chase.sh K G [VARIANT]}
VARIANT=${3:-full}

# One torch thread per process: 254 workers map cleanly onto 256 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/gnn

# --time-budget 85500 (= 23h45m) leaves ~75min margin under the 25h SLURM wall
# for the workers to stop and the summary to be written.
uv run python -u -m ai.cage.record_chase chase "$K" "$G" \
    --variant "$VARIANT" \
    --workers 254 \
    --time-budget 85500 \
    --out results/records
status=$?

if [ "$status" -ne 0 ]; then
    echo "==> record chase FAILED (k=$K g=$G variant=$VARIANT) exit=$status"
    exit "$status"
fi
echo "==> record chase done (k=$K g=$G variant=$VARIANT)"
