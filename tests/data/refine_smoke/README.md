# Refine Smoke Comparison

Compares bare random 3-regular graphs vs. graphs refined by the classical
`TabuRefiner` (no ML), on `g_target = 6` over 5 seeds.

## Setup

- 30-vertex random 3-regular graphs (`nx.random_regular_graph(3, 30, seed=...)`)
- `g_target = 6` (penalise cycles of length 3, 4, 5)
- Cost function: `short_cycle_cost` with default exponential weights
  (`w_c = 2^(g_target - c)`)
- TabuRefiner: `max_iter=100`, `sample_size=80`, `tenure=10`, no ML scoring

## Results

| seed | bare cost | refined cost | iterations | improvement |
|-----:|----------:|-------------:|-----------:|------------:|
|    0 |        28 |            0 |          4 |          28 |
|    1 |        24 |            0 |          4 |          24 |
|    2 |        26 |            0 |          5 |          26 |
|    3 |        16 |            0 |          4 |          16 |
|    4 |        44 |            0 |          6 |          44 |

## Verdict

Classical tabu reaches `cost = 0` (girth ≥ 6 on every seed) in 4–6 iterations
on these small inputs. The graphs are easy: 30-vertex random 3-regular graphs
have at most ~50 short cycles, and 2-switch space is small enough that the
greedy classical heuristic suffices.

## Limitations / what's missing

- **No GNN-tabu comparison.** Training a `MoveOracle` to score swaps requires
  a real run of `ai/cage/refine/train.py`; this smoke compares only bare vs.
  classical. The GNN scorer code paths are exercised by `test_refine_oracle.py`
  (train + save + reload), but not in a head-to-head comparison.
- **Trivially-solvable inputs.** Bigger lifts (n ≥ 100) and harder targets
  (g ≥ 8) are where the GNN is expected to earn its keep. This smoke
  intentionally stays small to keep iteration time low.
- **Reproducibility.** Run via:
  ```
  uv run python -c "..."  # see git history of this commit for the exact snippet
  ```
  The script is inline rather than a separate file because the comparison is
  one-shot, not a recurring benchmark.

The honest goal of this smoke is to verify that the `refine` module's plumbing
works end-to-end and that the classical baseline is sane. A proper GNN-vs-
classical comparison belongs in a separate run with real training compute.
