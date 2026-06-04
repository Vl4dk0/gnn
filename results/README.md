# results/ — In-Process Parallel Benchmarking Framework

`results/` is the evaluation and benchmarking layer for this project. It compares every cage-construction approach and every trained node-prediction model in a single run: models are imported directly (no Flask server required) and every trial fans out across CPU cores via `ProcessPoolExecutor`.

**What it benchmarks:**

| Benchmark | What is compared |
|---|---|
| `cage` | Cage-graph generators: `randomwalk`, `astar`, `bruteforce`, `rl`, `voltage` (variants: `Algebraic`, `GirthPredictor`, `TabuPredictor`), `voltage_rl`, `forge` (with ablations and hyperparameter sweeps) |
| `refine` | Tabu-refinement: classical vs. GNN-guided (`MoveOracle`) |
| `excision` | Tree-excision repair: classical vs. RL policy |
| `degree` | All trained degree-prediction GNN models (architectures: `gcn`, `gin`, `sage`, `gps`, `loopy`) |
| `min_cycle` | All trained minimum-cycle-length prediction GNN models |

---

## Quick start

```bash
# Smallest possible run — quick battery, all benchmarks
uv run python -m results.runner --quick

# Quick run of one benchmark
uv run python -m results.runner --benchmarks cage --quick

# Specific benchmarks, 3 seeds, 6 workers
uv run python -m results.runner --benchmarks cage,refine --seeds 3 --workers 6

# Scope to one approach on two (k,g) targets
uv run python -m results.runner --benchmarks cage --approaches forge --targets 3-6,4-6 --seeds 3

# Scope node-prediction benchmarks to one architecture
uv run python -m results.runner --benchmarks degree --approaches sage

# Tune cage search budget and step cap
uv run python -m results.runner --benchmarks cage --cage-budget 60 --cage-max-steps 500000
```

---

## All CLI flags

```
uv run python -m results.runner [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--benchmarks NAME,...` | `all` | Comma-separated benchmark names, or `all` |
| `--quick` | off | Use the small battery (fewer targets/graphs) |
| `--seeds N` | `1` | Number of seeds per task |
| `--seed-base N` | `0` | First seed; seeds run over `range(seed_base, seed_base+seeds)` |
| `--approaches NAME,...` | all | Restrict cage approaches (e.g. `forge`) or node architectures (e.g. `sage`) |
| `--targets k-g,...` | full grid | Restrict cage `(k,g)` targets, e.g. `3-6` or `3-6,4-6` |
| `--cage-budget SECS` | `20.0` | Time budget per cage trial in seconds |
| `--cage-max-steps N` | `200000` | Step cap per cage trial |
| `--workers N` | auto | Worker processes (default: `min(8, cpu-2)`) |
| `--task-timeout SECS` | `120.0` | Hard per-task wall-clock cap (SIGALRM); exceeded tasks are recorded as timeouts |
| `--out-root DIR` | `results/runs` | Root directory for run outputs |

---

## Output layout

Each run creates a timestamped directory under `results/runs/`:

```
results/runs/2026-05-30_14-32-00/
    results.jsonl     # one JSON line per TrialResult, written as tasks complete
    raw.json          # same data as a JSON array (written at end of run)
    summary.md        # aggregated markdown tables per benchmark, (k,g) matrices
    cage.csv          # one row per TrialResult for the cage benchmark
    degree.csv
    min_cycle.csv
    ...
```

`results.jsonl` is written incrementally, so partial results survive if the run is killed. `results/runs/` is git-ignored; see the PERUN section for how to retrieve a run from the cluster.

### summary.md

`summary.md` contains, per benchmark:

- A table of `approach`, `variant`, `n_trials`, `success_rate`, `mean_elapsed_s`, and any benchmark-specific metrics.
- For cage-like benchmarks (`cage`, `refine`, `excision`): two `(k,g)` x approach matrices — mean time-to-solve and mean found-graph size (`|V|/|E|`) across all approaches and targets. Rows are sorted by Moore bound so harder targets sit lower.

---

## Querying results

`results.query` reads a run's `results.jsonl`, filters, aggregates, and prints to stdout. The default run is the newest under `results/runs/`.

```bash
# List available runs
uv run python -m results.query --list-runs

# Everything about forge in the latest run (JSON, default)
uv run python -m results.query --approach forge

# Compare voltage variants on one target, human-readable table
uv run python -m results.query --benchmark cage --approach voltage --target 4-6 --format table

# Per-trial rows (unaggregated, e.g. to see each seed)
uv run python -m results.query --approach forge --target 4-5 --raw

# Query a specific run by timestamp
uv run python -m results.query --run 2026-05-31_01-32-27 --approach forge

# Markdown output (pipes into prose or docs)
uv run python -m results.query --benchmark cage --format md
```

Aggregated output groups by `(benchmark, approach, variant, target)`. Counts use all trials; mean time and size use only solved trials, so reported times are true time-to-solve figures. For cage benchmarks, compare `mean_nodes` against `moore_bound` (ratio 1.0 = a cage).

---

## Running on PERUN

Full benchmarks belong on PERUN (256-core CPU node). Scripts are in `supercomputer/`:

```bash
# Full benchmark — all five benchmarks, 5 seeds, 128 workers
sbatch supercomputer/run_benchmark.sh

# Forge-only benchmark — all forge variants, 40 seeds, 128 workers
sbatch supercomputer/run_benchmark_forge.sh
```

`run_benchmark.sh` runs:
```
uv run python -m results.runner \
    --benchmarks degree,min_cycle,cage,refine,excision \
    --seeds 5 --cage-budget 60 --cage-max-steps 200000 \
    --task-timeout 120 --workers 128
```

To retrieve a completed run, force-add its directory from PERUN, commit, push, then pull locally:
```bash
# on PERUN
git add -f results/runs/<stamp>
git commit -m "chore: add benchmark run <stamp>"
git push

# locally
git pull
```

---

## Adding a new benchmark

1. Create `results/benchmarks/my_benchmark.py` with two module-level callables and a `register` call:

```python
from results.registry import RunConfig, Task, Benchmark, register
from results.metrics import TrialResult

def make_tasks(config: RunConfig) -> list[Task]:
    # One Task per (instance x approach x variant).
    # Task.payload must contain only picklable primitives — no live graphs or tensors.
    ...

def execute(task: Task) -> list[TrialResult]:
    # Runs in a worker process; must be a module-level function.
    ...

register(Benchmark("my_benchmark", make_tasks, execute))
```

2. Add the import to `results/benchmarks/__init__.py`:

```python
from . import my_benchmark  # noqa: F401
```

3. Verify: `uv run python -m results.runner --benchmarks my_benchmark --quick`

---

## Design notes

- **CPU-only workers**: `_init_worker` sets `CUDA_VISIBLE_DEVICES=""` and `torch.set_num_threads(1)` so workers do not contend over cores or trigger MPS/CUDA + multiprocessing issues.
- **Hard timeouts**: each worker arms a `SIGALRM` for `--task-timeout` seconds. A runaway task is interrupted and recorded as a timeout rather than hanging the run.
- **Deterministic batteries**: `results/battery.py` provides seeded input sets (`node_battery`, `cage_targets`, `refine_instances`, `excision_instances`) shared across all benchmarks. The same `--quick` flag always produces the same inputs.
- **macOS spawn safety**: `Task.payload` carries only JSON-serialisable primitives. Live graphs and tensors are reconstructed inside `execute()` from seeds or names.
