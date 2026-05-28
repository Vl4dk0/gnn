# Evaluation Metrics Mismatch

**Source/Occurrence:** 
- `thesis/chapters/04-preparatory-tasks.typ` (Line 55)
- `ai/min_cycle/train.py` (Lines 158-173, `evaluate_model` function)

**Explanation:**
The thesis states: "the results are reported using exact rounded accuracy together with mean absolute error, root mean squared error, and the fraction of predictions that are off by one." 
However, the codebase implementation in `evaluate_model` calculates and reports only Mean Squared Error (`mse`), Mean Absolute Error (`mae`), and `accuracy`. It does not calculate the Root Mean Squared Error (`rmse`), nor does it compute the "fraction of predictions that are off by one" metric. Additionally, the MSE returned in the code is computed based on the rounded predictions rather than continuous predictions.

**Actionable Steps:**
1. Update `evaluate_model` in `ai/min_cycle/train.py` to calculate the Root Mean Squared Error (RMSE) instead of just returning MSE.
2. Add logic to calculate and return the "fraction of predictions that are off by one" metric (e.g., `(torch.abs(predictions_rounded - y) == 1).float().mean().item()`).
3. Ensure these metrics are properly returned and correctly logged during training and final evaluation, aligning with the thesis claims.
