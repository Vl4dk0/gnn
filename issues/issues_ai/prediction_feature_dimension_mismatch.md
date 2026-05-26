# Issue: GNN Feature Dimension Mismatch during Prediction for `degree` and `min_cycle` Tasks

## Severity
High (Causes immediate runtime crash when running inference using any GNN model trained with non-default node feature dimensions)

## Description
In `ai/degree/functions/graph_service.py` and `ai/min_cycle/functions/graph_service.py`, the `predict_all_nodes` function performs GNN-based inference to predict node degrees and minimal cycle lengths. During this process, the function builds the input node features `x` by concatenating a normalized node index, a random embedding, and a clustering coefficient placeholder:
```python
    x = torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)
```
This always results in a 4-dimensional node feature tensor (`x` of shape `[num_nodes, 4]`).

However, the GNN models for these tasks (GIN, GraphSAGE, GCN, GPS, and Loopy GNN) are trainable with configurable feature dimensions using the `--input-dim` command-line argument (which defaults to 4, but can be set to 1 for constant features, or any other integer).

If a model is trained with `--input-dim 1` (constant features), it is instantiated with `input_dim = 1`. During prediction, when `predict_all_nodes` runs, it will hardcode building a 4-dimensional feature tensor and pass it to the model. The model's first GNN layer or input linear projection (which expects input of size 1) receives input of size 4, resulting in a PyTorch runtime shape mismatch error (e.g. `RuntimeError: mat1 and mat2 shapes cannot be multiplied`).

The input features should be built dynamically depending on `model.input_dim`. If `model.input_dim == 1`, it should construct constant 1D node features (using `torch.ones`). If it is `4`, it should construct the default 4D features.

## Location
- `ai/degree/functions/graph_service.py` (lines 129-147)
- `ai/min_cycle/functions/graph_service.py` (lines 157-177)

## Proposed Fix
Modify `predict_all_nodes` in both files to check `model.input_dim` and construct the node features `x` accordingly:
```python
    if model.input_dim == 1:
        x = torch.ones((num_nodes, 1), dtype=torch.float)
    else:
        # Build 4-dimensional features
        node_idx_feature = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1) / max(num_nodes - 1, 1)
        
        _ = torch.manual_seed(42)
        random_feature = torch.randn(num_nodes, 2)
        
        _ = torch.manual_seed(42)
        clustering_feature = torch.rand(num_nodes, 1)
        
        x = torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)
```
