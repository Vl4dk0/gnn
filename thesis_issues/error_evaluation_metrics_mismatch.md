# Evaluation Metrics Mismatch in Minimum-Cycle Prediction

## Source of the issue
- `thesis/chapters/04-preparatory-tasks.typ` (Line 55)
- `ai/min_cycle/train.py` (Lines 158-173, `evaluate_model` function)

## Definition of the issue
The thesis specifies that evaluation results are reported using exact rounded accuracy, mean absolute error (MAE), root mean squared error (RMSE), and the fraction of predictions that are off by exactly one.

However, the evaluation implementation in `ai/min_cycle/train.py` currently computes and returns only Mean Squared Error (MSE), Mean Absolute Error (MAE), and Accuracy. It entirely omits the calculation of Root Mean Squared Error (RMSE) and the "fraction of predictions that are off by one" metric. Furthermore, the MSE it returns is computed using rounded integer predictions rather than the continuous raw outputs, creating a non-standard metric out of alignment with the training regression objective.

This implementation mismatch invalidates the reported evaluation metrics in the thesis. To resolve this, `evaluate_model` must be updated to correctly calculate the "off by one" metric (e.g., `(torch.abs(predictions_rounded - y) == 1).float().mean().item()`) and the proper RMSE (derived from the MSE of continuous outputs). These metrics must then be properly logged and included in the validation dictionaries to align with the thesis claims.
