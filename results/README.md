# results/ — In-Process Parallel Benchmarking Framework

Replaces the old monolithic HTTP-based `validation.py`.  Everything runs
**in-process** (models are imported directly, no Flask server required) and
**in parallel** across CPU cores via `ProcessPoolExecutor`.

---

## Quick start

```bash
# Run all benchmarks, small battery
uv run python -m results.runner --quick

# Run specific benchmarks only
uv run python -m results.runner --benchmarks cage,refine --quick

# Full run, 3 seeds, 6 workers
uv run python -m results.runner --seeds 3 --workers 6

# Tune cage search budget
uv run python -m results.runner --benchmarks cage --cage-budget 60 --cage-max-steps 500000

# Benchmark ONE approach on a couple of targets (small + fast, runs locally)
uv run python -m results.runner --benchmarks cage --approaches forge --targets 3-6,4-6 --seeds 3

# Benchmark a single node-prediction architecture
uv run python -m results.runner --benchmarks degree --approaches sage
```

`--approaches` restricts to the named approaches (cage: `randomwalk,astar,bruteforce,rl,voltage,voltage_rl,forge`; node tasks: the architecture, e.g. `sage,gin,gps`). `--targets` restricts the cage `(k,g)` grid as `k-g` pairs (e.g. `3-6,4-6`). Both default to "all", so you no longer have to run every approach just to test one.

Outputs are written to `results/runs/<YYYY-MM-DD_HH-MM-SS>/` (git-ignored).

---

## Output layout

```
results/runs/2026-05-30_14-32-00/
    raw.json          # list of serialised TrialResult dicts
    summary.md        # aggregated markdown table per benchmark
    degree.csv        # one row per TrialResult for the 'degree' benchmark
    cage.csv
    ...
```

---

## Metrics captured (`TrialResult`)

| Field | Meaning |
|---|---|
| `benchmark` | "degree"\|"min_cycle"\|"cage"\|"refine"\|"excision" |
| `approach` | construction/inference strategy used |
| `variant` | sub-variant (model arch, GNN vs classical, …) |
| `label` | human-readable unique identifier |
| `instance` | graph or target id |
| `target` | `{"k":…,"g":…}` or `{}` |
| `success` | whether the run produced a valid result |
| `elapsed_s` | wall time for this trial |
| `steps` | search steps / refine iterations (if applicable) |
| `n_nodes` / `n_edges` | result graph size |
| `metrics` | extensible dict: accuracy, mae, girth, moore_ratio, … |
| `model_id` / `model_params` / `model_size_mb` / `model_hparams` | model provenance |
| `error` | repr of exception if `success=False` |

---

## How to add a new benchmark

1. **Create** `results/benchmarks/my_benchmark.py`.

2. **Define** two module-level callables:

   ```python
   from results.registry import RunConfig, Task, Benchmark, register
   from results.metrics import TrialResult

   def make_tasks(config: RunConfig) -> list[Task]:
       # Build one Task per (instance × approach × variant).
       # Task.payload must contain ONLY picklable primitives.
       # Reconstruct graphs/tensors inside execute() from seeds or names.
       ...

   def execute(task: Task) -> list[TrialResult]:
       # Runs in a worker process — must be a module-level function.
       # Import heavy deps at the top of the module (not inside this function).
       ...

   register(Benchmark("my_benchmark", make_tasks, execute))
   ```

3. **Register** the import in `results/benchmarks/__init__.py`:

   ```python
   from . import my_benchmark  # noqa: F401
   ```

4. Run `uv run python -m results.runner --benchmarks my_benchmark --quick` to verify.

---

## Design notes

- **macOS spawn safety**: all worker entrypoints (`_execute`) are module-level.
  `Task.payload` carries only JSON-serialisable primitives — no live graphs or tensors.
- **CPU-only workers**: `_init_worker` disables CUDA/MPS and sets
  `torch.set_num_threads(1)` so workers don't fight over cores.
- **Deterministic batteries**: `results/battery.py` provides seeded graph sets
  (`node_battery`, `cage_targets`, `refine_instances`, `excision_instances`)
  shared across all benchmarks for reproducibility.
