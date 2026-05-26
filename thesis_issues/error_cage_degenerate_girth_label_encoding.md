# Infinite Girth Encoded as 0 in Training Data, Conflating Two Distinct Cases
## Source of the issue
- `ai/cage/voltage/data_gen.py` (Lines 70-76, `base_graph_to_pyg` function)
- `thesis/chapters/07-experiments-results.typ` (Lines 14-16)

## Definition of the issue
In `base_graph_to_pyg`, when `compute_lift_girth` returns `math.inf` (meaning the lift is degenerate/disconnected, or no closed identity-voltage walk was found within the maximum bounds), the code sets `girth_int = 0` and `girth_class = 0`. 

The issue is that this conflates two entirely different situations under the same numerical regression label `0`:
1. A valid graph that actually contains a cycle of length 0 (which is physically impossible, as cycles have length ≥ 1, with self-loops having length 1).
2. A degenerate or disconnected lift, or a tree, which has no cycles at all.

The thesis states that the predictor uses an auxiliary girth-regression output to learn a target girth value. By teaching the model that a degenerate or acyclic graph has a target girth of `0`, we introduce a fundamentally flawed supervision signal for the regression head. The model could incorrectly learn that a `0` target means "skip this" or become confused about the continuous scale of actual graph girths.

To resolve this issue, the data generation should assign a distinct out-of-range sentinel value (e.g., `-1`) for infinite-girth cases, and add an `is_degenerate` boolean flag to the generated `Data` object. Furthermore, during training, these degenerate samples should be masked from the regression loss so that the model is only penalized by the binary cross-entropy loss (via `girth_class = 0`) and not improperly penalized for failing to regress a meaningless target value.
