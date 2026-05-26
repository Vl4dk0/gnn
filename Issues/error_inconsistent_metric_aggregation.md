# Inconsistent Metric Aggregation (Graph-level vs Node-level)

## Source of the issue
- Codebase: `c:\Users\bskon\gnn\ai\min_cycle\train.py` (lines 158-172, `evaluate_model` function)
- Thesis: `c:\Users\bskon\gnn\thesis\chapters\04-preparatory-tasks.typ` (lines 55-61)

## Definition of the issue
In `evaluate_model`, there is a statistical inconsistency in how the metrics are aggregated across multiple graphs of varying sizes. The `accuracy` is computed at the node level by accumulating the total number of correct predictions divided by the total number of predictions (`total_correct / total_predictions`). However, `MSE` and `MAE` are computed by taking the mean error for each graph (`F.mse_loss` and `F.l1_loss` default to `reduction='mean'`), summing these graph-level means (`total_loss += loss.item()`), and then dividing by the number of graphs (`total_loss / num_test_graphs`). 

This gives equal weight to small and large graphs for MSE and MAE, while accuracy is correctly weighted by the number of vertices. Since the thesis specifies the metrics over the total 1790 vertices, all metrics (including MSE/RMSE and MAE) should be calculated globally at the node level (sum of all squared/absolute errors divided by the total number of vertices).

To fix this, the loss reduction for evaluation metrics should be changed to `'sum'` (e.g., `F.mse_loss(predictions, y, reduction='sum')` and `F.l1_loss(predictions_rounded, y, reduction='sum')`). The code should accumulate the sum of errors (`total_squared_error += loss.item()` and `total_absolute_error += mae.item()`) and divide the accumulated sums by `total_predictions` at the end to return the global node-level MSE/RMSE and MAE, matching the accuracy calculation.
