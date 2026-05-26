# MSE Evaluation Uses Rounded Predictions Instead of Continuous Output

## Source of the issue
- `ai/degree/train.py` (Lines 160-164, `evaluate_model` function)
- `ai/min_cycle/train.py` (Lines 154-158, `evaluate_model` function)
- `thesis/chapters/04-preparatory-tasks.typ` (Line 55)

## Definition of the issue
During model evaluation, the mean squared error (MSE) is computed using **rounded** predictions instead of the raw, continuous output:

```python
predictions_rounded = torch.round(out)
loss = F.mse_loss(predictions_rounded, y)
```

This means the "MSE" metric stored in the checkpoint `info.json` and reported during validation is the mean squared error of the rounded integer predictions against the true integer labels. Because the differences are all integers, this value is essentially a measure of the squared integer errors rather than the true regression error. 

The thesis explicitly states that the models "used a regression objective with mean squared error." During the training step, the loss correctly uses the continuous output (`F.mse_loss(out, y)`). Computing the evaluation MSE on rounded predictions causes a fundamental mismatch between the training objective and the evaluation metric. It also corrupts the Root Mean Squared Error (RMSE) values reported in the thesis, as calculating the square root of the MSE of rounded values produces a non-standard metric. 

To resolve this, the evaluation functions must be modified to compute the evaluation MSE (and consequently RMSE) using the continuous predictions `out`, aligning the reported evaluation metrics with the regression objective used during training.
