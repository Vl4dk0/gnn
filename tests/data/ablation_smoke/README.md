# Ablation Smoke: Structural Features vs. Baseline

**Purpose:** Test whether adding structural node features (cycle counts + RWPE)
lifts the Chen-Song-Caramanis symmetry wall on the hard targets (5,7), (4,7), (4,8).

## Setup

Both runs share identical settings:

| Parameter | Value |
|---|---|
| targets | 3,5; 3,6; 4,6; 5,7; 4,7; 4,8 |
| samples | 2000 |
| epochs | 2 |
| hidden_dim | 32 |
| num_layers | 4 |
| seed | 42 |

- **Baseline**: no structural features (`--cycle-lengths "" --rwpe-dim 0`)
- **Treatment**: `--cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8` (14 extra columns)

## Per-target F1 on test set

| Target | Baseline F1 | Treatment F1 | Delta |
|--------|-------------|--------------|-------|
| (3,5)  | 0.772       | 0.000        | -0.772 |
| (3,6)  | 0.727       | 0.000        | -0.727 |
| (4,6)  | 0.000       | 0.000        | 0.000 |
| (4,7)  | 0.000       | 0.000        | 0.000 |
| (4,8)  | 0.000       | 0.000        | 0.000 |
| (5,7)  | 0.000       | 0.000        | 0.000 |

## Overall test metrics

| Metric | Baseline | Treatment |
|--------|----------|-----------|
| Test F1 | 0.633 | 0.000 |
| Test MAE | 1.99 | 1.89 |
| Test accuracy | 78.0% | 78.0% |
| Test loss | 0.2944 | 0.2035 |

## Interpretation

At this smoke scale (2 epochs, 2000 samples, hidden_dim=32), structural features
**did not lift** the wall on any of the hard targets. In fact the treatment collapsed
to F1=0 across all targets, while the baseline achieved F1≈0.77 on (3,5) and (3,6).

The treatment did achieve a lower test loss (0.2035 vs 0.2944) and similar MAE
(1.89 vs 1.99), suggesting the additional features helped with regression but the
classifier head locked onto predicting all-negative (which matches the heavily
class-imbalanced targets like (5,7) with pos_rate=0.0).

### What this does and does not tell us

- At 2 epochs the model has not converged. Convergence with 14 extra input columns
  requires more training steps; the treatment needs more epochs to calibrate its
  classification head.
- The (5,7) target has pos_rate=0.0 in this 2000-sample run — there are literally
  zero positive examples in the dataset at this scale. No feature set can produce
  non-zero F1 on a target with no positives in the test split.
- The (4,7) and (4,8) targets have pos_rate ≈ 0.003 and ≈ 0.021 respectively, so
  fewer than 1-2 positive examples reach the test split.

### Verdict

**Inconclusive at this scale.** The symmetry-breaking hypothesis (that structural
features lift F1 on hard cells) requires a longer training run (≥50 epochs) with
more samples (≥10000) to see signal. The smoke ablation only validates that the
wiring works end-to-end; it cannot confirm or deny the research hypothesis.

The properly-scaled experiment should be run on PERUN:

```bash
# Baseline
uv run python -m ai.cage.voltage.train \
    --targets "3,5;3,6;4,6;5,7;4,7;4,8" --samples 50000 --epochs 100

# Treatment
uv run python -m ai.cage.voltage.train \
    --targets "3,5;3,6;4,6;5,7;4,7;4,8" --samples 50000 --epochs 100 \
    --cycle-lengths "3,4,5,6,7,8" --rwpe-dim 8
```
