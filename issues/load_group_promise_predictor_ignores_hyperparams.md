# Issue: load_group_promise_predictor Ignores Architectural Hyperparameters

## Severity
High

## Description
In `ai/cage/cayley/group_model.py`, the `load_group_promise_predictor` function reads only `hidden_dim` and `num_layers` from `info.json`. Other model shape parameters such as `node_feat_dim`, `edge_feat_dim`, `context_dim`, and `dropout` are initialized with hardcoded defaults.

If these hyperparameters diverged during training from their hardcoded defaults (for example, if a data generation change added a new node feature, or if the model was trained with a different number of layers/hidden dim defaults), `load_state_dict` will raise a shape mismatch runtime error during model loading without indicating what parameter is incorrect.

## Location
`ai/cage/cayley/group_model.py` (lines 170-173):
```python
    model = GroupPromisePredictor(
        hidden_dim=int(cast(int, training.get("hidden_dim", 96))),
        num_layers=int(cast(int, training.get("num_layers", 4))),
    )
```

## Proposed Fix
Save `node_feat_dim`, `edge_feat_dim`, `context_dim` to `info.json` during the training run and retrieve them when restoring the model in `load_group_promise_predictor`.
