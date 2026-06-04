# supercomputer/

SLURM batch scripts for the [PERUN HPC cluster](https://perun.tuke.sk) at TUKE.
Each file is a self-contained job submitted with `sbatch`.

---

## Submission workflow

1. **Connect to the TUKE VPN** (required before reaching the login node).

2. **SSH to the login node:**

   ```bash
   ssh <username>@login01.perun.tuke.sk
   ```

3. **Sync code** — push from your machine, then on PERUN:

   ```bash
   cd ~/gnn
   git pull
   ```

4. **Submit a job:**

   ```bash
   sbatch supercomputer/<script>.sh
   ```

   Some scripts accept positional arguments (see the script header for usage),
   e.g. `run_record_chase.sh`:

   ```bash
   sbatch --job-name=rc_8_5_full supercomputer/run_record_chase.sh 8 5 full
   ```

5. **Monitor:**

   ```bash
   squeue -u <username>          # list your jobs and their state
   ```

   Logs are written to `training_runs/` (GPU jobs) or `benchmark_runs/` (CPU
   jobs) under `~/gnn`, with filenames like `<job-name>_<jobid>.out` and
   `.err`.

6. **Retrieve results** — `results/runs/` and `results/records/` are
   gitignored.

   use scp or rsync to copy files from PERUN to your local machine, e.g.:

   ```bash
   scp <username>@login01.perun.tuke.sk:~/gnn/results/runs/<run-folder> ./local-folder
   ```

> **Always check the `.err` file about 1 minute after submission.** A job
> showing state `R` (running) in `squeue` may still have crashed immediately
> (bad import, missing weight file, etc.). Reading the `.err` file early saves
> wall-time.

---

## Key SBATCH header parameters

Every script carries a header block like the one below. Parameters marked with
`<...>` must be filled in with your cluster credentials before submitting.

```bash
#SBATCH --job-name=<descriptive-name>
#SBATCH --output=training_runs/%x_%j.out   # %x = job name, %j = job id
#SBATCH --error=training_runs/%x_%j.err
#SBATCH --partition=GPU                    # GPU or CPU (see below)
#SBATCH --account=<account>                # your PERUN allocation account
#SBATCH --qos=<account>                    # usually the same as --account
#SBATCH --gres=gpu:1                       # 1 GPU (GPU partition only)
#SBATCH --cpus-per-task=8                  # adjust per script
#SBATCH --mem=32G                          # adjust per script
#SBATCH --time=48:00:00                    # wall-time limit (max 48 h)
```

| Parameter | Typical values | Notes |
|---|---|---|
| `--partition` | `GPU` or `CPU` | Training jobs use `GPU`; benchmarks and record-chasing use `CPU` (256-core node) |
| `--account` / `--qos` | `<account>` | Both must match your allocation identifier |
| `--gres=gpu:1` | present on GPU jobs | Nodes carry NVIDIA H200 GPUs (143 GB VRAM) |
| `--time` | `06:00:00` – `48:00:00` | Long RL training caps at `48:00:00`; quick jobs (e.g. move-oracle) use 6 h |
| `--cpus-per-task` | `8` (GPU jobs), `256` (CPU jobs) | CPU partition jobs fan work out across all 256 cores |
| `--mem` | `32G` – `512G` | CPU partition jobs request `512G` for large parallel workloads |

