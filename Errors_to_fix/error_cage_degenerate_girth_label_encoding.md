# Infinite Girth Encoded as 0 in Training Data, Conflating Two Distinct Cases

**Source/Occurrence:**
- `ai/cage/voltage/data_gen.py` (Lines 70-76, `base_graph_to_pyg` function)
- `thesis/chapters/07-experiments-results.typ` (Lines 7-17)

**Explanation:**
In `base_graph_to_pyg`, a lift whose computed girth is `math.inf` (no short identity-voltage walk found) is encoded as `girth_int = 0` and `girth_class = 0`:

```python
if isinstance(girth, float):
    girth_int = 0
    girth_class = 0
```

This conflates two completely different situations under the same label `0`:
1. A lift where the smallest cycle has girth 0 (which is physically meaningless — a cycle must have length ≥ 1).
2. A degenerate lift (disconnected, or no identity-voltage closed walk found up to `max_girth = 2 * g_target`).

The model is trained to regress a target of `0` for degenerate lifts. However, the regression head (as described in the thesis, lines 14-16) is supposed to predict the girth. Teaching the model that "no cycle found" = "girth 0" introduces a fundamentally wrong supervision signal. The model may learn that girth-0 means "skip this", conflating it with the actual cycle of length 0 (a loop), or it may confuse the regression range entirely.

A better encoding would use a sentinel value clearly outside the valid girth range (e.g., `-1` or a separate boolean `is_degenerate` flag), and exclude such samples from the regression loss while still using the binary classification label.

**Actionable Steps:**
1. In `base_graph_to_pyg` in `ai/cage/voltage/data_gen.py`, change the degenerate encoding:
   - Set `girth_int = -1` (or another out-of-range sentinel) for infinite-girth (degenerate) cases.
   - Keep `girth_class = 0` for the binary classification head.
   - Add a `data.is_degenerate` flag set to `True` for these cases.
2. In the girth predictor training loop, mask degenerate samples from the regression loss (only apply the binary cross-entropy loss on them), so the model is not penalized for predicting any specific regression value for these inputs.
