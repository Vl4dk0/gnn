# Missing Evaluation Metrics in Degree Prediction

## Source of the issue
- `ai/degree/train.py` (Lines 135-180, `evaluate_model` function)
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 53-56, Section: Evaluation)

## Definition of the issue
The thesis specifies that results are reported using exact rounded accuracy, mean absolute error (MAE), root mean squared error (RMSE), and the fraction of predictions that are off by one. 

However, the `evaluate_model` function in `ai/degree/train.py` only calculates and returns mean squared error (MSE), MAE, and Accuracy. It does not calculate or return RMSE (though it returns MSE), and it completely omits the metric computing the fraction of predictions that are off by one. To ensure the implementation aligns with the thesis claims, the evaluation function needs to be updated. It should compute the fraction of predictions that are off by exactly one (`(torch.abs(predictions_rounded - y) == 1).sum().item() / y.numel()`), calculate the RMSE (the square root of the continuous MSE), and properly include these new metrics in the logging statements and the final metrics dictionary passed to `save_model`.
