# Missing Evaluation Metrics in Degree Prediction
## Source/Occurrence
- Codebase: `ai/degree/train.py` (`evaluate_model` function)
- Thesis: `thesis/chapters/04-preparatory-tasks.typ` (Section: Evaluation)

## Explanation
The thesis specifies that results are reported using exact rounded accuracy, mean absolute error (MAE), root mean squared error (RMSE), and the fraction of predictions that are off by one. However, the `evaluate_model` function in `ai/degree/train.py` only calculates and returns MSE, MAE, and Accuracy. It does not calculate or return RMSE (though it returns MSE) and it completely omits the "off by one" metric.

## Actionable steps
1. In `ai/degree/train.py`, update `evaluate_model` to calculate the fraction of predictions that are off by one: `off_by_one = (torch.abs(predictions_rounded - y) == 1).sum().item() / y.numel()`.
2. Update the `evaluate_model` return dictionary to include `"rmse"` (which can be calculated as the square root of the accumulated continuous MSE or rounded MSE, matching the thesis intent) and `"off_by_one"`.
3. Update the logging statements and final metric dictionary in `train_gnn` to print RMSE and the off-by-one fraction, and include them in the metrics dictionary passed to `save_model`.
