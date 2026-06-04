# ai/

This directory contains GNN model implementations, the shared model registry, and per-task training entrypoints. The three main prediction tasks are **degree** (node degree prediction), **min\_cycle** (shortest cycle through each node), and **cage** (constructing (k,g)-graphs of given degree and girth). Each task has its own training scripts; cage construction is further broken into several sub-tasks with independent trainers.

---

## Available models

Models live in `ai/models/` and are identified by short keys:

| Key | Class |
|-----|-------|
| `gcn` | `GCN_GNN` — Graph Convolutional Network |
| `sage` | `SAGE_GNN` — GraphSAGE with sum aggregation |
| `gin` | `GIN_GNN` — Graph Isomorphism Network |
| `loopy` | `Loopy_GNN` — r-ℓMPNN, cycle-aware (detects cycles up to r+2) |
| `gps` | `GPS_GNN` — GPS with transformer-style global attention |

---

## How to start training

All commands use the `uv run python -m ...` form. Run them from the project root.

### Degree prediction

Trains a node-level degree predictor on random Erdős-Rényi graphs.

```bash
uv run python -m ai.degree.train --model gin --epochs 5000
uv run python -m ai.degree.train --model loopy --r 3 --epochs 5000
```

### Min-cycle prediction

Trains a node-level predictor of the shortest cycle through each node.

```bash
uv run python -m ai.min_cycle.train --model gin --epochs 5000
uv run python -m ai.min_cycle.train --model loopy --r 3 --epochs 5000
```

### Shared flags — degree and min\_cycle

Both scripts accept the same set of flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` / `-m` | (required) | Model key: `gcn`, `sage`, `gin`, `loopy`, `gps` |
| `--name` / `-n` | auto-generated | Version label; the final model ID will be `<model>_<name>`. Supply this yourself. |
| `--epochs` | 5000 | Training epochs |
| `--hidden-dim` | 64 | Hidden layer width |
| `--num-layers` | 4 | Number of GNN message-passing layers |
| `--dropout` | 0.2 | Dropout probability |
| `--lr` | 0.001 | Adam learning rate |
| `--input-dim` | 4 | Node feature dimension (1 = constant, 4 = richer features) |
| `--graphs-per-epoch` | 10 | Random graphs trained on per epoch |
| `--eval-every` | 20 | Evaluation interval (epochs) |
| `--print-every` | 100 | Progress-print interval (epochs) |
| `--force` / `-f` | off | Overwrite an existing model with the same ID |
| `--r` | 3 | r-neighborhood radius for `loopy` (ignored otherwise) |
| `--conv-type` | `gin` | Inner conv for `gps`: `gcn`, `sage`, or `gin` |
| `--heads` | 4 | Attention heads for `gps` (ignored otherwise) |

### Cage — direct RL (PPO) over graph construction

```bash
uv run python -m ai.cage.rl.train --model gin --steps 200000
uv run python -m ai.cage.rl.train --model gin --steps 200000 --no-random
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` / `-m` | `gin` | Model key |
| `--name` / `-n` | auto-generated | Run label; supply yourself. |
| `--steps` | 200000 | Total environment steps (PPO) |
| `--hidden-dim` | 64 | Hidden dimension |
| `--num-layers` | 4 | GNN layers |
| `--dropout` | 0.2 | Dropout |
| `--lr` | 3e-4 | Learning rate |
| `--update-interval` | 2048 | Steps per PPO rollout buffer |
| `--episode-steps` | 500 | Max steps per episode |
| `--entropy-coef` | 0.05 | PPO entropy bonus coefficient |
| `--no-random` | off | Disable progressive curriculum randomization |
| `--force` / `-f` | off | Start from scratch; ignore existing checkpoint |
| `--r` | 3 | r-neighborhood radius for `loopy` |
| `--conv-type` | `gin` | Inner conv for `gps` |
| `--heads` | 4 | Attention heads for `gps` |
| `--log` | 3.0 | Live-log update interval in seconds (0 = disabled) |
| `--save` | 5 | Save a checkpoint every N completed episodes |

Without `--force`, the trainer **resumes** from an existing checkpoint of the same ID rather than overwriting it.

### Cage — voltage assignment RL

Trains a PPO agent to assign voltages to base graphs so that the voltage lift achieves target girth.

```bash
uv run python -m ai.cage.voltage.rl.train --steps 100000
uv run python -m ai.cage.voltage.rl.train --steps 200000 --hidden-dim 128
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 100000 | Total training steps |
| `--hidden-dim` | 64 | Hidden dimension |
| `--num-layers` | 3 | GINEConv layers |
| `--lr` | 3e-4 | Learning rate |
| `--update-interval` | 512 | PPO update interval |
| `--entropy-coef` | 0.05 | Entropy coefficient |
| `--conv-type` | `gine` | Convolution type: `gine` or `gps` |
| `--heads` | 4 | Attention heads (GPS only) |
| `--no-random` | off | Disable curriculum |
| `--log` | 3.0 | Live-log interval (seconds) |
| `--name` | `voltage_ppo` | Save name; supply your own value. |
| `--save-every` | 50000 | Periodic checkpoint interval in steps (0 = disabled) |

### Cage — voltage girth predictor

Always pass the full target grid via `--targets`.

```bash
uv run python -m ai.cage.voltage.supervised.train \
    --targets "3,5;3,6;3,7;4,5;4,6;5,5;5,6" --samples 200000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--targets` | (required) | Semicolon-separated `k,g` pairs for the training grid |
| `--samples` | 50000 | Number of training samples |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--hidden-dim` | 64 | Hidden dimension |
| `--num-layers` | 4 | GINEConv layers |
| `--lr` | 1e-3 | Learning rate |
| `--pos-weight` | 5.0 | Loss multiplier for the rare positive class |
| `--max-group-order` | 60 | Maximum voltage-group order |
| `--kind` | `girth` | Regression target: `girth` or `tabu_cost` |
| `--cycle-lengths` | `3,4,5,6,7,8` | Cycle-count features to include |
| `--rwpe-dim` | 8 | Random-walk positional encoding dimension (0 = off) |
| `--weight-decay` | 0.0 | AdamW weight decay |
| `--seed` | 42 | Random seed |
| `--workers` | auto | Data-generation worker count |

### Cage — GNN refinement oracle (MoveOracle)

Trains a GNN to predict the cost-delta of edge swaps for tabu search refinement.

```bash
uv run python -m ai.cage.refine.train \
    --samples 5000 --epochs 50 --batch-size 32 \
    --hidden-dim 64 --num-layers 3 --lr 1e-3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--samples` | 5000 | Training samples |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--hidden-dim` | 64 | Hidden dimension |
| `--num-layers` | 3 | GNN layers |
| `--lr` | 1e-3 | Learning rate |
| `--cycle-lengths` | `3,4,5,6,7,8` | Cycle-count structural features |
| `--rwpe-dim` | 8 | RWPE dimension |
| `--backbone` | `gine` | GNN backbone: `gine` or `loopy` |
| `--r` | 3 | r-neighborhood radius (loopy backbone only) |
| `--lift-fraction` | 0.7 | Fraction of samples from voltage-lift near-misses |
| `--val-fraction` | 0.1 | Held-out validation fraction |
| `--patience` | 8 | Early-stopping patience (epochs without improvement) |
| `--seed` | 42 | Random seed |
| `--workers` | auto | Data-generation workers |

### Cage — excision repair policy

Trains a PPO agent to repair degree-deficient graphs produced by BFS tree-excision.

```bash
uv run python -m ai.cage.excision.train --episodes 100 --g-target 5 --depth 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 100 | Total training episodes |
| `--g-target` | 5 | Target girth |
| `--depth` | 1 | BFS excision depth |
| `--hidden-dim` | 64 | Hidden dimension |
| `--cycle-lengths` | `3,4,5,6,7,8` | Structural cycle-count features |
| `--rwpe-dim` | 8 | RWPE dimension |
| `--instance-source` | `synthetic` | Repair-instance bank: `synthetic`, `lifts`, or `cages` |
| `--match-size` | 2 | Matching size removed to create an instance (difficulty) |
| `--num-instances` | 64 | Number of pre-generated instances |
| `--seed` | 42 | Random seed |

---

## Model registry and adding new models

The registry (`ai/registry.py`) discovers, loads, and saves trained models. It uses a hand-maintained mapping in `ai/models/__init__.py`:

```python
MODEL_CLASSES = {
    "gcn":   GCN_GNN,
    "sage":  SAGE_GNN,
    "gin":   GIN_GNN,
    "loopy": Loopy_GNN,
    "gps":   GPS_GNN,
}
```

To add a new model so the training scripts and registry can discover it:

1. **Create `ai/models/your_model.py`** and define a class that:
   - Subclasses `BaseGNN` (from `ai/models/base.py`)
   - Implements `forward(data: Data) -> torch.Tensor` (node-level output)
   - Passes `input_dim`, `hidden_dim`, `output_dim`, `num_layers`, `dropout` to `super().__init__()`
   - Overrides `get_config()` if it has additional hyperparameters that must be persisted across save/load

2. **Register it** in `ai/models/__init__.py`:
   ```python
   from .your_model import YourModel_GNN

   MODEL_CLASSES = {
       ...,
       "yourkey": YourModel_GNN,
   }
   ```

3. **Update `load_model`** in `ai/registry.py` if the model takes constructor arguments beyond the five base parameters (`input_dim`, `hidden_dim`, `output_dim`, `num_layers`, `dropout`). Add a branch analogous to the existing `loopy` and `gps` cases that reads extra fields from `training_config` and passes them to the constructor.

---

## Trained artifacts

Weights and metadata land under `ai/trained/<task>/<model_id>/`:

```
ai/trained/
  degree/
    gin_h128_l6/
      weights.pt
      info.json
  min_cycle/
    ...
  cage/
    ...
  excision/
    ...
  move_oracle/
    ...
  move_oracle_loopy_r3/
    ...
  voltage_girth/
    girth_predictor/
      weights.pt
      info.json
    tabu_predictor/
      weights.pt
      info.json
```

`info.json` records the model type, training hyperparameters, and evaluation metrics. The registry reads it on `load_model` to reconstruct the correct architecture before applying the saved weights.
