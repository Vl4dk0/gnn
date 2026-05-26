# Degree Model Checkpoint Saved with MSE from Rounded Predictions, Not Continuous Output

**Source/Occurrence:**
- `ai/degree/train.py` (Lines 160-164, `evaluate_model` function; Lines 360-366, checkpoint save)
- `thesis/chapters/04-preparatory-tasks.typ` (Line 55)

**Explanation:**
In `evaluate_model` for degree prediction, MSE is computed using **rounded** predictions:

```python
predictions_rounded = torch.round(out)
loss = F.mse_loss(predictions_rounded, y)
```

This means the "MSE" stored in the checkpoint `info.json` is actually the mean squared error of the **rounded integer** predictions against the true labels — which is equal to the fraction of wrong predictions (since rounded errors are always integers, MSE of integers vs integers is the mean of squared integer errors).

The thesis (line 55) states the early experiments "used a regression objective with mean squared error". The training loss itself uses the **continuous** output (`F.mse_loss(out, y)` in `train_step`). However, the logged and saved MSE metric comes from the rounded predictions, not the continuous regression output. This makes the logged "MSE" not a proper regression MSE and inconsistent with the training loss.

This also means the RMSE reported in the thesis tables (which says RMSE is computed and reported) would be `sqrt(MSE_of_rounded)` — an unusual, non-standard metric that is not the standard RMSE of a regressor.

**Actionable Steps:**
1. In `evaluate_model` in `ai/degree/train.py`, compute MSE separately from both the continuous output and the rounded output:
   - `mse_continuous = F.mse_loss(out, y)` — the standard regression MSE
   - `mse_rounded = F.mse_loss(predictions_rounded, y)` — for completeness
2. Log and save `mse_continuous` as the `"mse"` field so it matches the training loss objective.
3. RMSE should be derived from `mse_continuous`, not `mse_rounded`.
4. Apply the same fix to `ai/min_cycle/train.py`.
